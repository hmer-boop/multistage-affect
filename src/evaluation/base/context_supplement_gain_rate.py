# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metric_runtime_common import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    parse_valid_label_list,
    extract_stage_value,
    choose_candidate_ids,
)


STAGE2 = "stage2"
STAGE3 = "stage3"


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if "\n" in text and text.lstrip().startswith("{") and text.count("\n{") >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def records_to_id_map(records: Any) -> Dict[str, Dict[str, Any]]:
    """
    兼容以下常见格式：
    1) list[dict]
    2) dict[id] = dict
    3) {"valid": list[dict]}
    """
    id_map: Dict[str, Dict[str, Any]] = {}

    if isinstance(records, dict) and isinstance(records.get("valid"), list):
        records = records["valid"]

    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue
            item_id = rec.get("item_id") or rec.get("id")
            if item_id is None:
                continue
            id_map[str(item_id)] = rec
        return id_map

    if isinstance(records, dict):
        for k, v in records.items():
            if not isinstance(v, dict):
                continue
            rec = dict(v)
            rec["item_id"] = str(v.get("item_id") or v.get("id") or k)
            id_map[str(rec["item_id"])] = rec
        return id_map

    return id_map


def load_id_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return records_to_id_map(load_json_or_jsonl(path))


def merge_id_maps(*maps: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    合并同类文件，例如 base + round2。
    后传入的 map 覆盖前面的同 ID 记录，和当前 baseline 多数指标口径保持一致。
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for mp in maps:
        merged.update(mp)
    return merged


def merge_stage_maps(*maps: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    将 stage2 与 stage3 预测文件合并到同一条记录中。
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for mp in maps:
        for iid, rec in mp.items():
            merged.setdefault(iid, {})
            merged[iid].update(rec)
            merged[iid]["item_id"] = iid
    return merged


def labels_hit(pred_labels: Optional[List[str]], gold_labels: Optional[List[str]]) -> bool:
    """
    金标允许单标签或双标签。
    只要预测标签命中当前阶段金标集合中的任一标签，即视为该阶段正确。
    预测为空或非法时，parse_valid_label_list 会返回 None，按错误处理。
    """
    if not pred_labels or not gold_labels:
        return False
    return bool(set(pred_labels) & set(gold_labels))


def get_eval_ids(
    gold_map: Dict[str, Dict[str, Any]],
    pred23_map: Dict[str, Dict[str, Any]],
    eval_id_mode: str,
) -> List[str]:
    """
    9.6 只评价 Stage II -> Stage III，因此要求：
    1. gold 中 stage2 和 stage3 均为有效金标；
    2. pred 中同时存在 stage2 和 stage3 的原始输出字段。
    注意：预测字段存在但标签为空/非法时不跳过，而是在计算中按错误处理。
    """
    candidate_ids = choose_candidate_ids(gold_map, pred23_map, eval_id_mode)
    valid_ids: List[str] = []

    for iid in candidate_ids:
        grec = gold_map[iid]
        prec = pred23_map[iid]

        g2 = parse_valid_label_list(extract_stage_value(grec, STAGE2))
        g3 = parse_valid_label_list(extract_stage_value(grec, STAGE3))

        p2_raw = extract_stage_value(prec, STAGE2)
        p3_raw = extract_stage_value(prec, STAGE3)

        if g2 and g3 and p2_raw is not None and p3_raw is not None:
            valid_ids.append(iid)

    return valid_ids


def evaluate_context_supplement_gain(
    gold_map: Dict[str, Dict[str, Any]],
    pred23_map: Dict[str, Dict[str, Any]],
    eval_id_mode: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Context Supplement Gain Rate, CSGR:

        CSGR = #(Stage2 错误 且 Stage3 正确) / #(Stage2 错误)

    含义：
    - Stage2 错误：仅基于图像的延长观察阶段没有命中 stage2 金标；
    - Stage3 正确：引入标题/描述文本后命中 stage3 金标；
    - 该指标衡量文本语境是否帮助模型将 Stage2 的错误判断修正为 Stage3 的正确判断。
    """
    eval_ids = get_eval_ids(gold_map, pred23_map, eval_id_mode)

    stage2_wrong_count = 0
    context_gain_count = 0

    stage2_correct_count = 0
    stage3_correct_count = 0
    context_regression_count = 0  # Stage2 对，但 Stage3 错，仅作为辅助诊断

    detail_rows: List[Dict[str, Any]] = []

    for iid in eval_ids:
        grec = gold_map[iid]
        prec = pred23_map[iid]

        g2 = parse_valid_label_list(extract_stage_value(grec, STAGE2)) or []
        g3 = parse_valid_label_list(extract_stage_value(grec, STAGE3)) or []
        p2 = parse_valid_label_list(extract_stage_value(prec, STAGE2)) or []
        p3 = parse_valid_label_list(extract_stage_value(prec, STAGE3)) or []

        s2_correct = labels_hit(p2, g2)
        s3_correct = labels_hit(p3, g3)

        if s2_correct:
            stage2_correct_count += 1
        else:
            stage2_wrong_count += 1

        if s3_correct:
            stage3_correct_count += 1

        is_gain = (not s2_correct) and s3_correct
        is_regression = s2_correct and (not s3_correct)

        if is_gain:
            context_gain_count += 1
        if is_regression:
            context_regression_count += 1

        detail_rows.append({
            "item_id": iid,
            "gold_stage2": "|".join(g2),
            "pred_stage2": "|".join(p2),
            "stage2_correct": "1" if s2_correct else "0",
            "gold_stage3": "|".join(g3),
            "pred_stage3": "|".join(p3),
            "stage3_correct": "1" if s3_correct else "0",
            "context_gain": "1" if is_gain else "0",
            "context_regression": "1" if is_regression else "0",
        })

    total = len(eval_ids)
    csgr = context_gain_count / stage2_wrong_count if stage2_wrong_count else 0.0
    stage2_acc = stage2_correct_count / total if total else 0.0
    stage3_acc = stage3_correct_count / total if total else 0.0
    regression_rate = context_regression_count / stage2_correct_count if stage2_correct_count else 0.0

    summary = {
        "eval_total": total,
        "stage2_correct_count": stage2_correct_count,
        "stage3_correct_count": stage3_correct_count,
        "stage2_wrong_count": stage2_wrong_count,
        "context_gain_count": context_gain_count,
        "context_regression_count": context_regression_count,
        "stage2_acc_on_eval_ids": stage2_acc,
        "stage3_acc_on_eval_ids": stage3_acc,
        "context_supplement_gain_rate": csgr,
        "context_regression_rate": regression_rate,
    }

    return summary, detail_rows


def main():
    parser = build_common_parser("9.6 Context Supplement Gain Rate", include_output_csv=True)
    args = parser.parse_args()
    io = resolve_common_args(args, include_output_csv=True)

    check_required_files([io.gold_base, io.pred_stage2, io.pred_stage3])

    gold_map = merge_id_maps(
        load_id_map(io.gold_base),
        load_id_map(io.gold_round2),
    )

    pred2 = merge_id_maps(
        load_id_map(io.pred_stage2),
        load_id_map(io.pred_stage2_round2),
    )
    pred3 = merge_id_maps(
        load_id_map(io.pred_stage3),
        load_id_map(io.pred_stage3_round2),
    )

    # 只保留同时具有 stage2 与 stage3 预测记录的样本，避免单阶段未跑完时误入分母。
    common_pred_ids = sorted(set(pred2.keys()) & set(pred3.keys()))
    pred23_map = merge_stage_maps(
        {iid: pred2[iid] for iid in common_pred_ids},
        {iid: pred3[iid] for iid in common_pred_ids},
    )

    summary, rows = evaluate_context_supplement_gain(
        gold_map=gold_map,
        pred23_map=pred23_map,
        eval_id_mode=io.eval_id_mode,
    )

    if io.output_csv:
        io.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with io.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "item_id",
                    "gold_stage2",
                    "pred_stage2",
                    "stage2_correct",
                    "gold_stage3",
                    "pred_stage3",
                    "stage3_correct",
                    "context_gain",
                    "context_regression",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    print("===== 9.6 Context Supplement Gain Rate =====")
    print(f"参与样本数: {summary['eval_total']}")
    print(f"Stage2 正确数: {summary['stage2_correct_count']}")
    print(f"Stage2 错误数: {summary['stage2_wrong_count']}")
    print(f"Stage3 正确数: {summary['stage3_correct_count']}")
    print(f"语境补充增益数 Stage2错→Stage3对: {summary['context_gain_count']}")
    print(f"语境补充增益率: {summary['context_supplement_gain_rate']:.4f}")
    print(f"Stage2 对→Stage3 错数量: {summary['context_regression_count']}")
    print(f"Stage2 对→Stage3 错比例: {summary['context_regression_rate']:.4f}")


if __name__ == "__main__":
    main()
