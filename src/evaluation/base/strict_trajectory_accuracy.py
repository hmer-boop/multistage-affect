# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from metric_runtime_common import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
)


LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)
STAGES = ["stage1", "stage2", "stage3"]


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if "\n" in text and text.lstrip().startswith("{") and text.count("\n{") >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def rows_from_obj(obj: Any, path: Path) -> List[dict]:
    if isinstance(obj, dict) and isinstance(obj.get("valid"), list):
        rows = obj["valid"]
    elif isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = list(obj.values())
    else:
        raise ValueError(f"{path.name} 格式不符合预期：应为列表，或包含 valid 的字典。")
    return [r for r in rows if isinstance(r, dict)]


def _as_list(x: Any) -> List[str]:
    """
    与 9.1 TSMR 保持一致：
    - None/空字符串 -> []
    - str -> [str]
    - list -> 清洗后 list
    不额外按分隔符拆分，避免 STA 和 9.1 的 stage-level hit 口径不一致。
    """
    if x is None:
        return []
    if isinstance(x, list):
        out: List[str] = []
        for v in x:
            if isinstance(v, str):
                v = v.strip()
                if v:
                    out.append(v)
            elif v is not None:
                sv = str(v).strip()
                if sv:
                    out.append(sv)
        return out
    if isinstance(x, str):
        x = x.strip()
        return [x] if x else []
    sx = str(x).strip()
    return [sx] if sx else []


def first_pred_label(raw: Any) -> str:
    """
    与 9.1 TSMR 保持一致：
    预测若是 list，只取第一个标签；其余转成单个字符串；不做分隔符拆分。
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, str):
        return raw.strip()
    return ""


def load_gold_single(path: Path) -> Dict[str, Dict[str, List[str]]]:
    obj = load_json_or_jsonl(path)
    rows = rows_from_obj(obj, path)
    gold: Dict[str, Dict[str, List[str]]] = {}
    for r in rows:
        iid = r.get("item_id") or r.get("id")
        if not iid:
            continue
        iid = str(iid)
        gold[iid] = {
            "stage1": _as_list(r.get("stage1")),
            "stage2": _as_list(r.get("stage2")),
            "stage3": _as_list(r.get("stage3")),
        }
    return gold


def merge_gold_like_9_1(
    base_map: Dict[str, Dict[str, List[str]]],
    extra_map: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    与 9.1 保持一致：主 gold 优先；round2 只补充不存在的 ID，不覆盖。
    """
    merged = dict(base_map)
    for iid, info in extra_map.items():
        if iid in merged:
            continue
        merged[iid] = info
    return merged


def load_pred_one_stage_like_9_1(path: Path, stage: str) -> Dict[str, str]:
    obj = load_json_or_jsonl(path)
    rows = rows_from_obj(obj, path)
    pred: Dict[str, str] = {}
    stage_key1 = f"{stage}_label"
    stage_key2 = stage
    for r in rows:
        iid = r.get("item_id") or r.get("id")
        if not iid:
            continue
        lab = r.get(stage_key1)
        if lab is None:
            lab = r.get(stage_key2)
        pred[str(iid)] = first_pred_label(lab)
    return pred


def merge_pred_maps(*maps: Dict[str, str]) -> Dict[str, str]:
    """
    与 9.1 保持一致：后面的预测文件覆盖前面的同 ID 记录。
    """
    merged: Dict[str, str] = {}
    for mp in maps:
        merged.update(mp)
    return merged


def is_correct(pred_label: str, gold_labels: List[str]) -> bool:
    if pred_label not in LABEL_SET:
        return False
    return pred_label in gold_labels


def get_valid_trajectory_ids(
    gold: Dict[str, Dict[str, List[str]]],
    pred1: Dict[str, str],
    pred2: Dict[str, str],
    pred3: Dict[str, str],
    eval_id_mode: str,
) -> List[str]:
    """
    STA 分母：
    - gold 三阶段均非空；
    - eval_id_mode=pred 时，还要求三个阶段均存在预测记录，适合未跑完的模型；
    - eval_id_mode=gold 时，以 gold 三阶段非空样本为分母，缺失预测自然记为错误。
    """
    valid_gold_ids = [
        iid for iid, g in gold.items()
        if g.get("stage1") and g.get("stage2") and g.get("stage3")
    ]

    if eval_id_mode == "pred":
        common_pred_ids = set(pred1.keys()) & set(pred2.keys()) & set(pred3.keys())
        return [iid for iid in valid_gold_ids if iid in common_pred_ids]

    return valid_gold_ids


def evaluate_strict_trajectory_accuracy(
    gold: Dict[str, Dict[str, List[str]]],
    pred1: Dict[str, str],
    pred2: Dict[str, str],
    pred3: Dict[str, str],
    eval_id_mode: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    eval_ids = get_valid_trajectory_ids(gold, pred1, pred2, pred3, eval_id_mode)

    strict_correct_count = 0
    stage_correct_counts = {stage: 0 for stage in STAGES}
    detail_rows: List[Dict[str, Any]] = []

    for iid in eval_ids:
        g = gold[iid]
        p = {
            "stage1": pred1.get(iid, ""),
            "stage2": pred2.get(iid, ""),
            "stage3": pred3.get(iid, ""),
        }
        c = {
            "stage1": is_correct(p["stage1"], g["stage1"]),
            "stage2": is_correct(p["stage2"], g["stage2"]),
            "stage3": is_correct(p["stage3"], g["stage3"]),
        }

        for stage in STAGES:
            if c[stage]:
                stage_correct_counts[stage] += 1

        strict_correct = c["stage1"] and c["stage2"] and c["stage3"]
        if strict_correct:
            strict_correct_count += 1

        detail_rows.append({
            "item_id": iid,
            "gold_stage1": "|".join(g["stage1"]),
            "pred_stage1": p["stage1"],
            "stage1_correct": "1" if c["stage1"] else "0",
            "gold_stage2": "|".join(g["stage2"]),
            "pred_stage2": p["stage2"],
            "stage2_correct": "1" if c["stage2"] else "0",
            "gold_stage3": "|".join(g["stage3"]),
            "pred_stage3": p["stage3"],
            "stage3_correct": "1" if c["stage3"] else "0",
            "strict_trajectory_correct": "1" if strict_correct else "0",
        })

    total = len(eval_ids)
    summary = {
        "eval_total": total,
        "strict_correct_count": strict_correct_count,
        "strict_trajectory_accuracy": strict_correct_count / total if total else 0.0,
        "stage1_correct_count": stage_correct_counts["stage1"],
        "stage2_correct_count": stage_correct_counts["stage2"],
        "stage3_correct_count": stage_correct_counts["stage3"],
        "stage1_acc_on_sta_denominator": stage_correct_counts["stage1"] / total if total else 0.0,
        "stage2_acc_on_sta_denominator": stage_correct_counts["stage2"] / total if total else 0.0,
        "stage3_acc_on_sta_denominator": stage_correct_counts["stage3"] / total if total else 0.0,
    }

    return summary, detail_rows


def write_detail_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "gold_stage1", "pred_stage1", "stage1_correct",
        "gold_stage2", "pred_stage2", "stage2_correct",
        "gold_stage3", "pred_stage3", "stage3_correct",
        "strict_trajectory_correct",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_common_parser("9.7 Strict Trajectory Accuracy (STA)", include_output_csv=True)
    args = parser.parse_args()
    io = resolve_common_args(args, include_output_csv=True)

    check_required_files([io.gold_base, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gold_base = load_gold_single(io.gold_base)
    gold_extra = load_gold_single(io.gold_round2) if io.gold_round2 and io.gold_round2.exists() else {}
    gold_map = merge_gold_like_9_1(gold_base, gold_extra)

    p1_base = load_pred_one_stage_like_9_1(io.pred_stage1, "stage1") if io.pred_stage1 and io.pred_stage1.exists() else {}
    p2_base = load_pred_one_stage_like_9_1(io.pred_stage2, "stage2") if io.pred_stage2 and io.pred_stage2.exists() else {}
    p3_base = load_pred_one_stage_like_9_1(io.pred_stage3, "stage3") if io.pred_stage3 and io.pred_stage3.exists() else {}

    p1_extra = load_pred_one_stage_like_9_1(io.pred_stage1_round2, "stage1") if io.pred_stage1_round2 and io.pred_stage1_round2.exists() else {}
    p2_extra = load_pred_one_stage_like_9_1(io.pred_stage2_round2, "stage2") if io.pred_stage2_round2 and io.pred_stage2_round2.exists() else {}
    p3_extra = load_pred_one_stage_like_9_1(io.pred_stage3_round2, "stage3") if io.pred_stage3_round2 and io.pred_stage3_round2.exists() else {}

    pred1 = merge_pred_maps(p1_base, p1_extra)
    pred2 = merge_pred_maps(p2_base, p2_extra)
    pred3 = merge_pred_maps(p3_base, p3_extra)

    summary, rows = evaluate_strict_trajectory_accuracy(
        gold=gold_map,
        pred1=pred1,
        pred2=pred2,
        pred3=pred3,
        eval_id_mode=io.eval_id_mode,
    )

    if io.output_csv:
        write_detail_csv(io.output_csv, rows)

    print("===== 9.7 Strict Trajectory Accuracy (STA) =====")
    print(f"评估口径: eval_id_mode={io.eval_id_mode}")
    if io.eval_id_mode == "pred":
        print("分母定义: gold 三阶段均非空，且三个阶段均存在预测记录")
    else:
        print("分母定义: gold 三阶段均非空；缺失预测记为错误")
    print(f"参与样本数: {summary['eval_total']}")
    print(f"严格轨迹命中数: {summary['strict_correct_count']}")
    print(f"stage1 同分母命中率: {summary['stage1_acc_on_sta_denominator']:.4f}")
    print(f"stage2 同分母命中率: {summary['stage2_acc_on_sta_denominator']:.4f}")
    print(f"stage3 同分母命中率: {summary['stage3_acc_on_sta_denominator']:.4f}")
    print(f"Strict Trajectory Accuracy (STA): {summary['strict_trajectory_accuracy']:.4f}")
    if io.output_csv:
        print(f"明细 CSV: {io.output_csv}")


if __name__ == "__main__":
    main()
