# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from metric_runtime_common_cot import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    load_json_or_jsonl,
    normalize_text,
    print_metric_io,
    extract_stage_label,
    merge_pred_maps,
)


LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)
SPLITTERS = ["/", "、", "|", ",", "，", ";", "；"]
STAGE2 = "stage2"
STAGE3 = "stage3"


def to_list_labels(x: Any, normalize_text) -> List[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        if "labels" in x:
            return to_list_labels(x.get("labels"), normalize_text)
        for key in [
            "label", "pred_label", "prediction_label", "emotion_label", "emotion",
            "pred", "prediction", "answer", "final_label", "result", "output",
        ]:
            if key in x:
                return to_list_labels(x.get(key), normalize_text)
        return []
    if isinstance(x, list):
        out: List[str] = []
        for v in x:
            out.extend(to_list_labels(v, normalize_text))
        return out
    s = normalize_text(x)
    if not s:
        return []
    parts = [s]
    for sp in SPLITTERS:
        new_parts = []
        for seg in parts:
            new_parts.extend(seg.split(sp))
        parts = new_parts
    return [normalize_text(v) for v in parts if normalize_text(v)]


def records_to_id_map(records: Any) -> Dict[str, Dict[str, Any]]:
    id_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(records, dict):
        if isinstance(records.get("valid"), list):
            records = records["valid"]
        elif isinstance(records.get("results"), list):
            records = records["results"]
        elif isinstance(records.get("data"), list):
            records = records["data"]

    if isinstance(records, dict):
        for k, v in records.items():
            if isinstance(v, dict):
                v2 = dict(v)
                v2["item_id"] = str(v.get("item_id") or v.get("id") or k)
                id_map[str(v2["item_id"])] = v2
        return id_map

    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue
            item_id = rec.get("item_id") or rec.get("id")
            if item_id is None:
                continue
            id_map[str(item_id)] = rec
    return id_map


def load_gold_valid_records(path, load_json_or_jsonl) -> Dict[str, Dict[str, Any]]:
    raw = load_json_or_jsonl(path)
    return records_to_id_map(raw)


def merge_gold_maps(gold_main: Dict[str, Dict[str, Any]], gold_round2: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    与 CoT 9.6 保持一致：主 gold 优先，round2 仅补充不存在的 ID。
    """
    merged = dict(gold_main)
    for item_id, rec in gold_round2.items():
        if item_id in merged:
            continue
        merged[item_id] = rec
    return merged


def load_pred_stage(path, stage_name: str, load_json_or_jsonl) -> Dict[str, Dict[str, Any]]:
    raw = load_json_or_jsonl(path)
    id_map = records_to_id_map(raw)
    pred_map: Dict[str, Dict[str, Any]] = {}
    for iid, rec in id_map.items():
        pred_map[iid] = {stage_name: extract_stage_label(rec, stage_name)}
    return pred_map


def legal_labels(labels: List[str]) -> List[str]:
    return [lb for lb in labels if lb in LABEL_SET]


def labels_hit(pred_labels: List[str], gold_labels: List[str]) -> bool:
    """
    金标允许单标签或双标签；预测命中任一金标即为该阶段正确。
    非法/空预测按错误处理。
    """
    p = set(legal_labels(pred_labels))
    g = set(legal_labels(gold_labels))
    return bool(p) and bool(g) and bool(p & g)


def evaluate_context_regression_rate(gold_map, pred2, pred3) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Context Regression Rate, CRR:

        CRR = #(Stage2 正确 且 Stage3 错误) / #(Stage2 正确)

    该指标只比较 Stage II 与 Stage III。越低越好。
    """
    common_pred_ids = sorted(set(pred2.keys()) & set(pred3.keys()))

    eval_total = 0
    stage2_correct_count = 0
    stage2_wrong_count = 0
    stage3_correct_count = 0
    context_gain_count = 0
    context_regression_count = 0
    detail_rows: List[Dict[str, Any]] = []

    for item_id in common_pred_ids:
        if item_id not in gold_map:
            continue

        grec = gold_map[item_id]
        g2 = legal_labels(to_list_labels(extract_stage_label(grec, STAGE2), normalize_text))
        g3 = legal_labels(to_list_labels(extract_stage_label(grec, STAGE3), normalize_text))
        if not g2 or not g3:
            continue

        p2 = legal_labels(to_list_labels(extract_stage_label(pred2[item_id], STAGE2), normalize_text))
        p3 = legal_labels(to_list_labels(extract_stage_label(pred3[item_id], STAGE3), normalize_text))

        eval_total += 1
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

    csgr = context_gain_count / stage2_wrong_count if stage2_wrong_count else 0.0
    crr = context_regression_count / stage2_correct_count if stage2_correct_count else 0.0

    summary = {
        "eval_total": eval_total,
        "stage2_correct_count": stage2_correct_count,
        "stage2_wrong_count": stage2_wrong_count,
        "stage3_correct_count": stage3_correct_count,
        "context_gain_count": context_gain_count,
        "context_regression_count": context_regression_count,
        "stage2_acc_on_eval_ids": stage2_correct_count / eval_total if eval_total else 0.0,
        "stage3_acc_on_eval_ids": stage3_correct_count / eval_total if eval_total else 0.0,
        "context_supplement_gain_rate": csgr,
        "context_regression_rate": crr,
    }
    return summary, detail_rows


def main():
    parser = build_common_parser("9.8 Context Regression Rate (CRR)")
    args = parser.parse_args()
    io = resolve_common_args(args)
    print_metric_io(io, quiet=getattr(args, "quiet_paths", False))
    required = [io.gold_base, io.gold_round2, io.pred_stage2, io.pred_stage3]
    required += [p for p in [io.pred_stage2_round2, io.pred_stage3_round2] if p]
    check_required_files(required)

    gold_map = merge_gold_maps(
        load_gold_valid_records(io.gold_base, load_json_or_jsonl),
        load_gold_valid_records(io.gold_round2, load_json_or_jsonl),
    )
    pred2 = merge_pred_maps(
        load_pred_stage(io.pred_stage2, STAGE2, load_json_or_jsonl),
        load_pred_stage(io.pred_stage2_round2, STAGE2, load_json_or_jsonl) if io.pred_stage2_round2 else {},
    )
    pred3 = merge_pred_maps(
        load_pred_stage(io.pred_stage3, STAGE3, load_json_or_jsonl),
        load_pred_stage(io.pred_stage3_round2, STAGE3, load_json_or_jsonl) if io.pred_stage3_round2 else {},
    )

    summary, _ = evaluate_context_regression_rate(gold_map, pred2, pred3)

    print("===== 9.8 Context Regression Rate (CRR) =====")
    print("分母定义: Stage2 正确样本数；分子定义: Stage2 正确且 Stage3 错误")
    print(f"参与样本数: {summary['eval_total']}")
    print(f"Stage2 正确数: {summary['stage2_correct_count']}")
    print(f"Stage2 错误数: {summary['stage2_wrong_count']}")
    print(f"Stage3 正确数: {summary['stage3_correct_count']}")
    print(f"语境补充增益数 Stage2错→Stage3对: {summary['context_gain_count']}")
    print(f"语境补充增益率 CSGR: {summary['context_supplement_gain_rate']:.4f}")
    print(f"语境回退数 Stage2对→Stage3错: {summary['context_regression_count']}")
    print(f"Context Regression Rate (CRR): {summary['context_regression_rate']:.4f}")


if __name__ == "__main__":
    main()
