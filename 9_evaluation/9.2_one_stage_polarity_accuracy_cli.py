from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from metric_runtime_common import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    parse_valid_label_list,
    extract_stage_value,
    choose_candidate_ids,
)

STAGES = ["stage1", "stage2", "stage3"]
POS_SET = {"宁静", "平静", "安宁", "恬静", "快乐", "喜悦", "高兴", "愉悦", "幸福", "惊奇", "惊讶", "敬畏", "崇敬", "肃然起敬"}
NEG_SET = {"悲伤", "伤感", "忧伤", "恐惧", "害怕", "畏惧", "厌恶", "反感", "恶心", "愤怒", "生气", "恼怒", "气愤"}
CANON_POS = {"宁静", "快乐", "惊奇", "敬畏"}
CANON_NEG = {"悲伤", "恐惧", "厌恶", "愤怒"}


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if "\n" in text and text.lstrip().startswith("{") and text.count("\n{") >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def label_to_polarity(label: str) -> Optional[str]:
    if not label:
        return None
    if label in POS_SET or label in CANON_POS:
        return "pos"
    if label in NEG_SET or label in CANON_NEG:
        return "neg"
    return None


def labels_to_polarity_set(labels: List[str]) -> set:
    pols = set()
    for lb in labels:
        pol = label_to_polarity(lb)
        if pol:
            pols.add(pol)
    return pols


def records_to_id_map(records: Any) -> Dict[str, Dict[str, Any]]:
    id_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(records, dict):
        for k, v in records.items():
            if isinstance(v, dict):
                v2 = dict(v)
                v2["item_id"] = k
                id_map[str(k)] = v2
        return id_map
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue
            item_id = rec.get("item_id") or rec.get("id")
            if not item_id:
                continue
            id_map[str(item_id)] = rec
    return id_map


def merge_pred_maps(*maps: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for mp in maps:
        for iid, rec in mp.items():
            merged.setdefault(iid, {}).update(rec)
    return merged


def get_valid_stage_ids(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], eval_id_mode: str, stage: str) -> List[str]:
    candidate_ids = choose_candidate_ids(gold_map, pred_map, eval_id_mode)
    valid_ids: List[str] = []
    for iid in candidate_ids:
        grec = gold_map[iid]
        prec = pred_map[iid]
        g_labels = parse_valid_label_list(extract_stage_value(grec, stage))
        p_labels = parse_valid_label_list(extract_stage_value(prec, stage))
        if g_labels and p_labels:
            valid_ids.append(iid)
    return valid_ids


def evaluate(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], eval_id_mode: str):
    per_stage_total = {s: 0 for s in STAGES}
    per_stage_exact_correct = {s: 0 for s in STAGES}
    per_stage_correct = {s: 0 for s in STAGES}
    stage_valid_ids = {s: get_valid_stage_ids(gold_map, pred_map, eval_id_mode, s) for s in STAGES}
    detail_rows = []

    for st in STAGES:
        for item_id in stage_valid_ids[st]:
            grec = gold_map[item_id]
            prec = pred_map[item_id]
            g_labels = parse_valid_label_list(extract_stage_value(grec, st)) or []
            p_labels = parse_valid_label_list(extract_stage_value(prec, st)) or []
            gold_pols = labels_to_polarity_set(g_labels)
            pred_pols = labels_to_polarity_set(p_labels)
            exact_correct = bool(set(g_labels) & set(p_labels))
            is_correct = bool(gold_pols & pred_pols)
            per_stage_total[st] += 1
            if exact_correct:
                per_stage_exact_correct[st] += 1
            if is_correct:
                per_stage_correct[st] += 1
            if exact_correct and not is_correct:
                raise AssertionError(
                    f"{st} polarity accuracy invariant failed for item_id={item_id}: "
                    f"gold_labels={g_labels}, pred_labels={p_labels}"
                )
            detail_rows.append({
                "item_id": item_id,
                "stage": st,
                "gold_labels": "|".join(g_labels),
                "gold_polarity": "|".join(sorted(gold_pols)) or "",
                "gpt_label": "|".join(p_labels),
                "gpt_polarity": "|".join(sorted(pred_pols)) or "",
                "correct_by_polarity": "1" if is_correct else "0",
            })

    per_stage_acc = {s: (per_stage_correct[s] / per_stage_total[s]) if per_stage_total[s] else 0.0 for s in STAGES}
    return {
        "stage_valid_counts": {s: len(stage_valid_ids[s]) for s in STAGES},
        "per_stage_total": per_stage_total,
        "per_stage_exact_correct": per_stage_exact_correct,
        "per_stage_correct": per_stage_correct,
        "per_stage_accuracy": per_stage_acc,
    }, detail_rows


def main():
    parser = build_common_parser("9.2 One-stage Polarity Accuracy", include_output_csv=True)
    args = parser.parse_args()
    io = resolve_common_args(args, include_output_csv=True)

    check_required_files([io.gold_base, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gold_raw = load_json_or_jsonl(io.gold_base)
    gold_raw_round2 = load_json_or_jsonl(io.gold_round2) if io.gold_round2 and io.gold_round2.exists() else []
    gold_map = {}
    gold_map.update(records_to_id_map(gold_raw))
    gold_map.update(records_to_id_map(gold_raw_round2))

    pred_base = merge_pred_maps(
        records_to_id_map(load_json_or_jsonl(io.pred_stage1)) if io.pred_stage1 and io.pred_stage1.exists() else {},
        records_to_id_map(load_json_or_jsonl(io.pred_stage2)) if io.pred_stage2 and io.pred_stage2.exists() else {},
        records_to_id_map(load_json_or_jsonl(io.pred_stage3)) if io.pred_stage3 and io.pred_stage3.exists() else {},
    )
    pred_round2 = merge_pred_maps(
        records_to_id_map(load_json_or_jsonl(io.pred_stage1_round2)) if io.pred_stage1_round2 and io.pred_stage1_round2.exists() else {},
        records_to_id_map(load_json_or_jsonl(io.pred_stage2_round2)) if io.pred_stage2_round2 and io.pred_stage2_round2.exists() else {},
        records_to_id_map(load_json_or_jsonl(io.pred_stage3_round2)) if io.pred_stage3_round2 and io.pred_stage3_round2.exists() else {},
    )
    pred_map = {}
    pred_map.update(pred_base)
    pred_map.update(pred_round2)

    summary, rows = evaluate(gold_map, pred_map, io.eval_id_mode)

    if io.output_csv:
        io.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with io.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["item_id", "stage", "gold_labels", "gold_polarity", "gpt_label", "gpt_polarity", "correct_by_polarity"])
            writer.writeheader()
            writer.writerows(rows)

    print("===== 9.2 Stage Polarity Accuracy =====")
    print(
        "参与样本数: "
        f"stage1={summary['stage_valid_counts']['stage1']}, "
        f"stage2={summary['stage_valid_counts']['stage2']}, "
        f"stage3={summary['stage_valid_counts']['stage3']}"
    )
    print(
        "标签命中数(校验): "
        f"stage1={summary['per_stage_exact_correct']['stage1']}, "
        f"stage2={summary['per_stage_exact_correct']['stage2']}, "
        f"stage3={summary['per_stage_exact_correct']['stage3']}"
    )
    print(
        "极性命中数: "
        f"stage1={summary['per_stage_correct']['stage1']}, "
        f"stage2={summary['per_stage_correct']['stage2']}, "
        f"stage3={summary['per_stage_correct']['stage3']}"
    )
    print(f"stage1: {summary['per_stage_accuracy']['stage1']:.4f}")
    print(f"stage2: {summary['per_stage_accuracy']['stage2']:.4f}")
    print(f"stage3: {summary['per_stage_accuracy']['stage3']:.4f}")


if __name__ == "__main__":
    main()
