from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

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


def is_stage_polarity_exact_match(gold_labels: List[str], pred_labels: List[str]) -> Tuple[bool, set, set]:
    gold_pols = labels_to_polarity_set(gold_labels)
    pred_pols = labels_to_polarity_set(pred_labels)
    return gold_pols == pred_pols, gold_pols, pred_pols


def get_valid_eval_ids(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], eval_id_mode: str) -> List[str]:
    candidate_ids = choose_candidate_ids(gold_map, pred_map, eval_id_mode)
    valid_ids: List[str] = []
    for iid in candidate_ids:
        grec = gold_map[iid]
        prec = pred_map[iid]
        ok = True
        for st in STAGES:
            g_labels = parse_valid_label_list(extract_stage_value(grec, st))
            p_labels = parse_valid_label_list(extract_stage_value(prec, st))
            if not g_labels or not p_labels:
                ok = False
                break
        if ok:
            valid_ids.append(iid)
    return valid_ids


def evaluate(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], eval_id_mode: str):
    valid_ids = get_valid_eval_ids(gold_map, pred_map, eval_id_mode)
    detail_rows = []
    trajectory_total = 0
    trajectory_exact_correct = 0

    for item_id in valid_ids:
        grec = gold_map[item_id]
        prec = pred_map[item_id]
        flags = []
        row = {"item_id": item_id}
        for st in STAGES:
            g_labels = parse_valid_label_list(extract_stage_value(grec, st)) or []
            p_labels = parse_valid_label_list(extract_stage_value(prec, st)) or []
            matched, gold_pols, pred_pols = is_stage_polarity_exact_match(g_labels, p_labels)
            flags.append(matched)
            row[f"{st}_gold_polarity"] = "|".join(sorted(gold_pols))
            row[f"{st}_gpt_polarity"] = "|".join(sorted(pred_pols))
            row[f"{st}_exact_match"] = "1" if matched else "0"
        trajectory_total += 1
        trajectory_exact = all(flags)
        if trajectory_exact:
            trajectory_exact_correct += 1
        row["trajectory_polarity_exact_match"] = "1" if trajectory_exact else "0"
        detail_rows.append(row)

    rate = trajectory_exact_correct / trajectory_total if trajectory_total else 0.0
    return {"valid_id_count": len(valid_ids), "rate": rate}, detail_rows


def main():
    parser = build_common_parser("9.2 Three-stage Polarity All Consistency", include_output_csv=True)
    args = parser.parse_args()
    io = resolve_common_args(args, include_output_csv=True)

    check_required_files([io.gold_base, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gold_map = {}
    gold_map.update(records_to_id_map(load_json_or_jsonl(io.gold_base)))
    if io.gold_round2 and io.gold_round2.exists():
        gold_map.update(records_to_id_map(load_json_or_jsonl(io.gold_round2)))

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
            writer = csv.DictWriter(f, fieldnames=["item_id", "stage1_gold_polarity", "stage1_gpt_polarity", "stage1_exact_match", "stage2_gold_polarity", "stage2_gpt_polarity", "stage2_exact_match", "stage3_gold_polarity", "stage3_gpt_polarity", "stage3_exact_match", "trajectory_polarity_exact_match"])
            writer.writeheader()
            writer.writerows(rows)

    print("===== 9.2 3stage_polarity_all_consistency =====")
    print(f"参与样本数: {summary['valid_id_count']}")
    print(f"3stage_polarity_all_consistency: {summary['rate']:.4f}")


if __name__ == "__main__":
    main()
