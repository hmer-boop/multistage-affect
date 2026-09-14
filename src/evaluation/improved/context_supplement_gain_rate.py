# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from metric_runtime_common_improve import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    load_json_or_jsonl,
    rows_from_obj,
    split_labels,
)

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)
SPLITTERS = ["/", "、", "|", ",", "，", ";", "；"]
STAGE2 = "stage2"
STAGE3 = "stage3"


def parse_valid_label_list(x: Any) -> Optional[List[str]]:
    """
    解析 gold / pred 标签字段。
    - 支持 str / list / {"labels": [...]}；
    - 支持常见分隔符；
    - 若为空或包含非法标签，则返回 None。
    """
    labels = split_labels(x, SPLITTERS)
    if not labels:
        return None
    return labels if all(lb in LABEL_SET for lb in labels) else None


def extract_stage_label(record: Dict[str, Any], stage: str) -> Any:
    """兼容 stage / stage_label / 其他包含 stage 的字段名。"""
    if not record:
        return None
    for key in (stage, f"{stage}_label"):
        if key in record:
            return record[key]
    for key in record.keys():
        if stage in key:
            return record[key]
    return None


def records_to_id_map(records: Any) -> Dict[str, Dict[str, Any]]:
    """兼容 list / dict / {"valid": [...]} 三类结构。"""
    if isinstance(records, dict) and isinstance(records.get("valid"), list):
        records = records["valid"]

    id_map: Dict[str, Dict[str, Any]] = {}

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


def load_id_map(path) -> Dict[str, Dict[str, Any]]:
    obj = load_json_or_jsonl(path)
    return records_to_id_map(obj)


def merge_gold_maps(gold_main: Dict[str, Dict[str, Any]], gold_round2: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    与 improve 原始指标口径保持一致：主 gold 优先，round2 只补充不存在的 ID，不覆盖。
    """
    merged = dict(gold_main)
    for item_id, rec in gold_round2.items():
        if item_id in merged:
            continue
        merged[item_id] = rec
    return merged


def load_gold_valid_records(path) -> Dict[str, Dict[str, Any]]:
    raw = load_json_or_jsonl(path)
    if isinstance(raw, dict) and isinstance(raw.get("valid"), list):
        raw = raw["valid"]
    return records_to_id_map(raw)


def load_pred_stage(path, stage_name: str) -> Dict[str, Dict[str, Any]]:
    """
    读取 improve 当前套评估显式传入的该阶段预测文件。
    注意：不读取 Base / CoT / improved 路径，不做额外路径推断。
    """
    raw = load_json_or_jsonl(path)
    rows = rows_from_obj(raw, path)
    pred_map: Dict[str, Dict[str, Any]] = {}
    for rec in rows:
        item_id = rec.get("id") or rec.get("item_id")
        if item_id is None:
            continue
        pred_map[str(item_id)] = {stage_name: extract_stage_label(rec, stage_name)}
    return pred_map


def labels_hit(pred_labels: Optional[List[str]], gold_labels: Optional[List[str]]) -> bool:
    """
    金标允许单标签或双标签。
    只要预测标签命中当前阶段金标集合中的任一标签，即视为当前阶段正确。
    预测为空或非法标签时按错误处理。
    """
    if not pred_labels or not gold_labels:
        return False
    return bool(set(pred_labels) & set(gold_labels))


def evaluate_context_supplement_gain(
    gold_map: Dict[str, Dict[str, Any]],
    pred2: Dict[str, Dict[str, Any]],
    pred3: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    语境补充增益率 Context Supplement Gain Rate, CSGR:

        CSGR = #(Stage2 错误 且 Stage3 正确) / #(Stage2 错误)

    该指标只比较 Stage II 与 Stage III：
    - Stage II 表示仅基于图像的延长视觉观察；
    - Stage III 表示引入标题/描述文本后的文本辅助解释；
    - 若 Stage II 错误且 Stage III 正确，则记为一次语境补充增益。
    """
    common_pred_ids = sorted(set(pred2.keys()) & set(pred3.keys()))

    eval_total = 0
    stage2_correct_count = 0
    stage3_correct_count = 0
    stage2_wrong_count = 0
    context_gain_count = 0
    context_regression_count = 0  # Stage2 对 -> Stage3 错，仅作辅助观察
    detail_rows: List[Dict[str, Any]] = []

    for item_id in common_pred_ids:
        if item_id not in gold_map:
            continue

        grec = gold_map[item_id]
        g2 = parse_valid_label_list(extract_stage_label(grec, STAGE2))
        g3 = parse_valid_label_list(extract_stage_label(grec, STAGE3))

        # 金标必须在 Stage2 和 Stage3 都有效；预测为空/非法不跳过，后续按错误处理。
        if not g2 or not g3:
            continue

        p2 = parse_valid_label_list(extract_stage_label(pred2[item_id], STAGE2)) or []
        p3 = parse_valid_label_list(extract_stage_label(pred3[item_id], STAGE3)) or []

        s2_correct = labels_hit(p2, g2)
        s3_correct = labels_hit(p3, g3)

        eval_total += 1

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
            "item_id": item_id,
            "gold_stage2": "|".join(g2),
            "pred_stage2": "|".join(p2),
            "stage2_correct": "1" if s2_correct else "0",
            "gold_stage3": "|".join(g3),
            "pred_stage3": "|".join(p3),
            "stage3_correct": "1" if s3_correct else "0",
            "context_gain": "1" if is_gain else "0",
            "context_regression": "1" if is_regression else "0",
        })

    context_supplement_gain_rate = context_gain_count / stage2_wrong_count if stage2_wrong_count else 0.0
    stage2_acc_on_eval_ids = stage2_correct_count / eval_total if eval_total else 0.0
    stage3_acc_on_eval_ids = stage3_correct_count / eval_total if eval_total else 0.0
    context_regression_rate = context_regression_count / stage2_correct_count if stage2_correct_count else 0.0

    summary = {
        "eval_total": eval_total,
        "stage2_correct_count": stage2_correct_count,
        "stage3_correct_count": stage3_correct_count,
        "stage2_wrong_count": stage2_wrong_count,
        "context_gain_count": context_gain_count,
        "context_regression_count": context_regression_count,
        "stage2_acc_on_eval_ids": stage2_acc_on_eval_ids,
        "stage3_acc_on_eval_ids": stage3_acc_on_eval_ids,
        "context_supplement_gain_rate": context_supplement_gain_rate,
        "context_regression_rate": context_regression_rate,
    }
    return summary, detail_rows


def main():
    parser = build_common_parser("9.6 Context Supplement Gain Rate")
    args = parser.parse_args()
    io = resolve_common_args(args)

    check_required_files([io.gold_base, io.gold_round2, io.pred_stage2, io.pred_stage3])

    gold_map = merge_gold_maps(
        load_gold_valid_records(io.gold_base),
        load_gold_valid_records(io.gold_round2),
    )
    pred2 = load_pred_stage(io.pred_stage2, STAGE2)
    pred3 = load_pred_stage(io.pred_stage3, STAGE3)

    summary, _ = evaluate_context_supplement_gain(gold_map, pred2, pred3)

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
