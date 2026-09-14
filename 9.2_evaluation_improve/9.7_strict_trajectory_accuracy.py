# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

from metric_runtime_common_improve import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    load_json_or_jsonl,
    rows_from_obj,
)


LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)
STAGES = ["stage1", "stage2", "stage3"]


def _as_list(x) -> List[str]:
    """
    与 improve 9.1 的 TSMR 保持一致：
    - None/空字符串 -> []
    - str -> [str]
    - list -> 清洗后 list
    不额外按分隔符拆分。
    """
    if x is None:
        return []
    if isinstance(x, list):
        out = []
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


def load_gold_single(path) -> Dict[str, Dict[str, List[str]]]:
    obj = load_json_or_jsonl(path)
    rows = rows_from_obj(obj, path)
    gold = {}
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


def merge_gold(base_map, extra_map):
    """
    与 improve 9.1 保持一致：主 gold 优先，round2 只补充不存在的 ID，不覆盖。
    """
    merged = dict(base_map)
    for iid, info in extra_map.items():
        if iid in merged:
            continue
        merged[iid] = info
    return merged


def _first_label(raw) -> str:
    """
    与 improve 9.1 保持一致：
    预测若是 list，只取第一个标签；其余转成单个字符串。
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, str):
        return raw.strip()
    return ""


def load_pred_stage(path, stage: str) -> Dict[str, str]:
    obj = load_json_or_jsonl(path)
    rows = rows_from_obj(obj, path)
    pred = {}
    key1 = f"{stage}_label"
    key2 = stage
    for r in rows:
        iid = r.get("id") or r.get("item_id")
        if not iid:
            continue
        lab = r.get(key1)
        if lab is None:
            lab = r.get(key2)
        pred[str(iid)] = _first_label(lab)
    return pred


def is_correct(pred_label: str, gold_labels: List[str]) -> bool:
    return pred_label in LABEL_SET and pred_label in gold_labels


def get_valid_sta_ids(gold, pred1, pred2, pred3) -> List[str]:
    """
    improve STA 分母与 improve 9.1 TSMR 对齐：
    三阶段都有预测，且 gold 三阶段均非空。
    """
    common_pred_ids = sorted(set(pred1.keys()) & set(pred2.keys()) & set(pred3.keys()))
    valid_ids = []
    for iid in common_pred_ids:
        if iid not in gold:
            continue
        g = gold[iid]
        if g.get("stage1") and g.get("stage2") and g.get("stage3"):
            valid_ids.append(iid)
    return valid_ids


def eval_sta(gold, pred1, pred2, pred3) -> Tuple[int, int, float, Dict[str, float], Dict[str, int]]:
    valid_ids = get_valid_sta_ids(gold, pred1, pred2, pred3)
    total = len(valid_ids)

    strict_correct_count = 0
    stage_correct_counts = {stage: 0 for stage in STAGES}

    for iid in valid_ids:
        g = gold[iid]
        c1 = is_correct(pred1.get(iid, ""), g["stage1"])
        c2 = is_correct(pred2.get(iid, ""), g["stage2"])
        c3 = is_correct(pred3.get(iid, ""), g["stage3"])

        if c1:
            stage_correct_counts["stage1"] += 1
        if c2:
            stage_correct_counts["stage2"] += 1
        if c3:
            stage_correct_counts["stage3"] += 1
        if c1 and c2 and c3:
            strict_correct_count += 1

    stage_acc = {
        stage: stage_correct_counts[stage] / total if total else 0.0
        for stage in STAGES
    }
    sta = strict_correct_count / total if total else 0.0
    return total, strict_correct_count, sta, stage_acc, stage_correct_counts


def main():
    parser = build_common_parser("9.7 Strict Trajectory Accuracy (STA)")
    args = parser.parse_args()
    io = resolve_common_args(args)
    check_required_files([io.gold_base, io.gold_round2, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gold = merge_gold(load_gold_single(io.gold_base), load_gold_single(io.gold_round2))
    pred1 = load_pred_stage(io.pred_stage1, "stage1")
    pred2 = load_pred_stage(io.pred_stage2, "stage2")
    pred3 = load_pred_stage(io.pred_stage3, "stage3")

    total, strict_correct_count, sta, stage_acc, stage_correct_counts = eval_sta(gold, pred1, pred2, pred3)

    print("===== 9.7 Strict Trajectory Accuracy (STA) =====")
    print("分母定义: 三阶段都有预测，且 gold 三阶段均非空")
    print(f"参与样本数: {total}")
    print(f"严格轨迹命中数: {strict_correct_count}")
    print(f"stage1 同分母命中率: {stage_acc['stage1']:.4f} | 命中数={stage_correct_counts['stage1']}")
    print(f"stage2 同分母命中率: {stage_acc['stage2']:.4f} | 命中数={stage_correct_counts['stage2']}")
    print(f"stage3 同分母命中率: {stage_acc['stage3']:.4f} | 命中数={stage_correct_counts['stage3']}")
    print(f"Strict Trajectory Accuracy (STA): {sta:.4f}")


if __name__ == "__main__":
    main()
