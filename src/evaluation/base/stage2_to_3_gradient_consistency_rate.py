# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Set

from metric_runtime_common import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    parse_valid_label_list,
    extract_stage_value,
    choose_candidate_ids,
)

POS = {"宁静", "快乐", "惊奇", "敬畏"}
NEG = {"悲伤", "恐惧", "厌恶", "愤怒"}


def polarity(label: str):
    if label in POS:
        return "POS"
    if label in NEG:
        return "NEG"
    return None


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if "\n" in text and text.lstrip().startswith("{") and text.count("\n{") >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_json_records(path: Path) -> List[Dict]:
    obj = load_json_or_jsonl(path)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "valid" in obj and isinstance(obj["valid"], list):
            return obj["valid"]
        return [v for v in obj.values() if isinstance(v, dict)]
    raise TypeError(f"{path} 顶层既不是 list 也不是 dict")


def normalize_gold_row(row: Dict) -> Dict:
    return {
        "item_id": row.get("item_id"),
        "stage1": parse_valid_label_list(row.get("stage1")) or [],
        "stage2": parse_valid_label_list(row.get("stage2")) or [],
        "stage3": parse_valid_label_list(row.get("stage3")) or [],
    }


def normalize_pred_row(row: Dict) -> Dict:
    out = {"item_id": row.get("item_id") or row.get("id")}
    for st in ["stage1", "stage2", "stage3"]:
        v = parse_valid_label_list(extract_stage_value(row, st))
        if v:
            out[st] = v
    return out


def index_by_id(rows: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for r in rows:
        iid = r.get("item_id")
        if iid is not None:
            out[str(iid)] = r
    return out


def merge_stage_records(*rows_groups: List[Dict]) -> List[Dict]:
    pred_map = {}
    for rows in rows_groups:
        mp = index_by_id(rows)
        for iid, rec in mp.items():
            pred_map.setdefault(iid, {"item_id": iid})
            for k, v in rec.items():
                if k == "item_id":
                    continue
                if isinstance(v, list) and len(v) > 0:
                    pred_map[iid][k] = v
                elif v not in (None, "", []):
                    pred_map[iid][k] = v
    return list(pred_map.values())


def merge_id_maps(*maps: Dict[str, Dict]) -> Dict[str, Dict]:
    merged = {}
    for mp in maps:
        for iid, rec in mp.items():
            merged[iid] = rec
    return merged


def polarity_set(labels: List[str]) -> Set[str]:
    out = set()
    for lb in labels:
        p = polarity(lb)
        if p is not None:
            out.add(p)
    return out


def grad_set_between(labels_a: List[str], labels_b: List[str]) -> Set[int]:
    pa_set = polarity_set(labels_a)
    pb_set = polarity_set(labels_b)
    if not pa_set or not pb_set:
        return set()
    out = set()
    for pa in pa_set:
        for pb in pb_set:
            if pa == pb:
                out.add(0)
            elif pa == "NEG" and pb == "POS":
                out.add(+1)
            elif pa == "POS" and pb == "NEG":
                out.add(-1)
    return out


def grad_match(g_a: List[str], g_b: List[str], p_a: List[str], p_b: List[str]):
    gg = grad_set_between(g_a, g_b)
    pp = grad_set_between(p_a, p_b)
    valid = len(gg) > 0 and len(pp) > 0
    ok = 1 if (valid and len(gg & pp) > 0) else 0
    return valid, ok, gg, pp


def get_valid_eval_ids(gold: Dict[str, Dict], pred: Dict[str, Dict], eval_id_mode: str) -> List[str]:
    candidate_ids = choose_candidate_ids(gold, pred, eval_id_mode)
    valid_ids: List[str] = []
    for iid in candidate_ids:
        g = gold[iid]
        p = pred[iid]
        if g.get("stage2") and g.get("stage3") and p.get("stage2") and p.get("stage3"):
            valid_ids.append(iid)
    return valid_ids


def main():
    parser = build_common_parser("9.5 stage2->3 gradient consistency rate", include_strict=True)
    args = parser.parse_args()
    io = resolve_common_args(args, include_strict=True)

    check_required_files([io.gold_base, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gold_rows_base = [normalize_gold_row(r) for r in load_json_records(io.gold_base) if isinstance(r, dict)]
    gold_rows_round2 = [normalize_gold_row(r) for r in load_json_records(io.gold_round2) if isinstance(r, dict)] if io.gold_round2 and io.gold_round2.exists() else []
    gold = merge_id_maps(index_by_id(gold_rows_base), index_by_id(gold_rows_round2))

    pred_rows_base = merge_stage_records(
        [normalize_pred_row(r) for r in load_json_records(io.pred_stage1) if isinstance(r, dict)] if io.pred_stage1 and io.pred_stage1.exists() else [],
        [normalize_pred_row(r) for r in load_json_records(io.pred_stage2) if isinstance(r, dict)] if io.pred_stage2 and io.pred_stage2.exists() else [],
        [normalize_pred_row(r) for r in load_json_records(io.pred_stage3) if isinstance(r, dict)] if io.pred_stage3 and io.pred_stage3.exists() else [],
    )
    pred_rows_round2 = merge_stage_records(
        [normalize_pred_row(r) for r in load_json_records(io.pred_stage1_round2) if isinstance(r, dict)] if io.pred_stage1_round2 and io.pred_stage1_round2.exists() else [],
        [normalize_pred_row(r) for r in load_json_records(io.pred_stage2_round2) if isinstance(r, dict)] if io.pred_stage2_round2 and io.pred_stage2_round2.exists() else [],
        [normalize_pred_row(r) for r in load_json_records(io.pred_stage3_round2) if isinstance(r, dict)] if io.pred_stage3_round2 and io.pred_stage3_round2.exists() else [],
    )
    pred = merge_id_maps(index_by_id(pred_rows_base), index_by_id(pred_rows_round2))

    valid_ids = get_valid_eval_ids(gold, pred, io.eval_id_mode)

    total = 0
    g23_correct = 0
    for iid in valid_ids:
        g = gold[iid]
        p = pred[iid]
        g2, g3 = g.get("stage2", []), g.get("stage3", [])
        p2, p3 = p.get("stage2", []), p.get("stage3", [])
        g23_valid, g23_ok, _, _ = grad_match(g2, g3, p2, p3)
        if not io.strict and not g23_valid:
            continue
        total += 1
        if g23_ok:
            g23_correct += 1

    rate = g23_correct / total if total else 0.0

    print("===== 9.5_stage2_to_3_gradient_consistency_rate =====")
    print(f"参与样本数: {len(valid_ids) if io.strict else total}")
    print(f"stage2_to_3_gradient_consistency_rate: {rate:.4f}")


if __name__ == "__main__":
    main()
