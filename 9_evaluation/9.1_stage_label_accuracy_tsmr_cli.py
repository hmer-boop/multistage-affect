# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from metric_runtime_common import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
)

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)
STAGES = ["stage1", "stage2", "stage3"]


def load_json_or_jsonl(path: Path):
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


def _as_list(x) -> List[str]:
    """严格沿用旧脚本：None/'' -> []; str -> [str]; list -> 清洗后 list。"""
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


def first_pred_label(raw: Any) -> str:
    """严格沿用旧脚本：list 只取第一个；其余转成单个字符串；不做分隔符拆分。"""
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


def merge_gold_like_old(base_map: Dict[str, Dict[str, List[str]]], extra_map: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    """严格沿用旧脚本：优先保留主 gold，补充不存在的 ID，不覆盖。"""
    merged = dict(base_map)
    for iid, info in extra_map.items():
        if iid in merged:
            continue
        merged[iid] = info
    return merged


def load_pred_one_stage_old(path: Path, stage: str) -> Dict[str, str]:
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
    merged: Dict[str, str] = {}
    for mp in maps:
        merged.update(mp)
    return merged


def is_correct(pred_label: str, gold_labels: List[str]) -> bool:
    if pred_label not in LABEL_SET:
        return False
    return pred_label in gold_labels


def evaluate_stage_by_pred_ids(gold: Dict[str, Dict[str, List[str]]], pred: Dict[str, str], stage: str) -> tuple[int, int]:
    """三个阶段统一按预测 ID 触发：
    - 先遍历 pred 的 id
    - gold 不存在则跳过
    - gold 当前阶段为空则跳过
    - 其余进入分母
    """
    eval_total = 0
    correct = 0
    for iid in pred.keys():
        if iid not in gold:
            continue
        gold_labels = gold[iid].get(stage, [])
        if not gold_labels:
            continue
        eval_total += 1
        if is_correct(pred.get(iid, ""), gold_labels):
            correct += 1
    return eval_total, correct


def evaluate_tsmr_old(
    gold: Dict[str, Dict[str, List[str]]],
    pred1: Dict[str, str],
    pred2: Dict[str, str],
    pred3: Dict[str, str],
) -> tuple[int, float]:
    """保留旧 TSMR：只保留 gold 三阶段都非空的共同样本集。"""
    valid_ids = [
        iid for iid, g in gold.items()
        if g.get("stage1") and g.get("stage2") and g.get("stage3")
    ]

    if not valid_ids:
        return 0, 0.0

    tsmr_sum = 0.0
    for iid in valid_ids:
        g = gold[iid]
        c1 = is_correct(pred1.get(iid, ""), g["stage1"])
        c2 = is_correct(pred2.get(iid, ""), g["stage2"])
        c3 = is_correct(pred3.get(iid, ""), g["stage3"])
        tsmr_sum += (int(c1) + int(c2) + int(c3)) / 3.0

    return len(valid_ids), tsmr_sum / len(valid_ids)


def main():
    parser = build_common_parser("9.1 Stage Label Accuracy + TSMR")
    args = parser.parse_args()
    io = resolve_common_args(args)

    check_required_files([io.gold_base, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    # ===== gold：严格按旧逻辑补充不存在的 ID，不覆盖主 gold =====
    gold_base = load_gold_single(io.gold_base)
    gold_extra = load_gold_single(io.gold_round2) if io.gold_round2 and io.gold_round2.exists() else {}
    gold_map = merge_gold_like_old(gold_base, gold_extra)

    # ===== pred：支持现有命名与可选额外文件，但单条样本解析严格按旧逻辑 =====
    p1_base = load_pred_one_stage_old(io.pred_stage1, "stage1") if io.pred_stage1 and io.pred_stage1.exists() else {}
    p2_base = load_pred_one_stage_old(io.pred_stage2, "stage2") if io.pred_stage2 and io.pred_stage2.exists() else {}
    p3_base = load_pred_one_stage_old(io.pred_stage3, "stage3") if io.pred_stage3 and io.pred_stage3.exists() else {}

    p1_extra = load_pred_one_stage_old(io.pred_stage1_round2, "stage1") if io.pred_stage1_round2 and io.pred_stage1_round2.exists() else {}
    p2_extra = load_pred_one_stage_old(io.pred_stage2_round2, "stage2") if io.pred_stage2_round2 and io.pred_stage2_round2.exists() else {}
    p3_extra = load_pred_one_stage_old(io.pred_stage3_round2, "stage3") if io.pred_stage3_round2 and io.pred_stage3_round2.exists() else {}

    pred1 = merge_pred_maps(p1_base, p1_extra)
    pred2 = merge_pred_maps(p2_base, p2_extra)
    pred3 = merge_pred_maps(p3_base, p3_extra)

    # ===== 三个阶段统一改为：按预测 ID 回查 gold =====
    stage1_total, stage1_correct = evaluate_stage_by_pred_ids(gold_map, pred1, "stage1")
    stage2_total, stage2_correct = evaluate_stage_by_pred_ids(gold_map, pred2, "stage2")
    stage3_total, stage3_correct = evaluate_stage_by_pred_ids(gold_map, pred3, "stage3")

    # ===== TSMR 先保留旧口径 =====
    tsmr_total, tsmr = evaluate_tsmr_old(gold_map, pred1, pred2, pred3)

    stage1_acc = (stage1_correct / stage1_total) if stage1_total else 0.0
    stage2_acc = (stage2_correct / stage2_total) if stage2_total else 0.0
    stage3_acc = (stage3_correct / stage3_total) if stage3_total else 0.0

    overall_total = stage1_total + stage2_total + stage3_total
    overall_correct = stage1_correct + stage2_correct + stage3_correct
    overall_acc = (overall_correct / overall_total) if overall_total else 0.0

    print("===== 9.1 Stage Label Accuracy + TSMR =====")
    print(
        "参与样本数: "
        f"stage1={stage1_total}, "
        f"stage2={stage2_total}, "
        f"stage3={stage3_total}, "
        f"tsmr={tsmr_total}"
    )
    print(f"stage1: {stage1_acc:.4f}")
    print(f"stage2: {stage2_acc:.4f}")
    print(f"stage3: {stage3_acc:.4f}")
    print(f"整体累计准确率: {overall_acc:.4f}")
    print(f"Trajectory segmentation matching: {tsmr:.4f}")


if __name__ == "__main__":
    main()
