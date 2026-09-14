# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Set

from metric_runtime_common_improve import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    load_json_or_jsonl,
    normalize_text,
)


from typing import Any, Dict, List, Optional

POS_SET = {"宁静", "平静", "安宁", "恬静", "快乐", "喜悦", "高兴", "愉悦", "幸福", "惊奇", "惊讶", "敬畏", "崇敬", "肃然起敬"}
NEG_SET = {"悲伤", "伤感", "忧伤", "恐惧", "害怕", "畏惧", "厌恶", "反感", "恶心", "愤怒", "生气", "恼怒", "气愤"}
CANON_POS = {"宁静", "快乐", "惊奇", "敬畏"}
CANON_NEG = {"悲伤", "恐惧", "厌恶", "愤怒"}
SPLITTERS = ["/", "、", "|", ",", "，", ";", "；"]


def to_list_labels(x: Any, normalize_text) -> List[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        if 'labels' in x:
            return to_list_labels(x.get('labels'), normalize_text)
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


def label_to_polarity(label: str) -> Optional[str]:
    if not label:
        return None
    if label in POS_SET or label in CANON_POS:
        return 'pos'
    if label in NEG_SET or label in CANON_NEG:
        return 'neg'
    return None


def labels_to_polarity_set(labels: List[str]) -> set:
    return {pol for pol in (label_to_polarity(lb) for lb in labels) if pol}


def extract_stage_label(record: Dict[str, Any], stage: str) -> Any:
    if not record:
        return None
    preferred = [stage, f'{stage}_label']
    for key in preferred:
        if key in record:
            return record[key]
    for k in record.keys():
        if stage in k:
            return record[k]
    return None


def records_to_id_map(records: Any) -> Dict[str, Dict[str, Any]]:
    id_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(records, dict):
        for k, v in records.items():
            if isinstance(v, dict):
                v2 = dict(v)
                v2['item_id'] = str(v.get('item_id') or v.get('id') or k)
                id_map[str(v2['item_id'])] = v2
        return id_map
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue
            item_id = rec.get('item_id') or rec.get('id')
            if item_id is None:
                continue
            id_map[str(item_id)] = rec
    return id_map


def load_gold_valid_records(path, load_json_or_jsonl):
    raw = load_json_or_jsonl(path)
    if isinstance(raw, dict) and isinstance(raw.get('valid'), list):
        raw = raw['valid']
    return records_to_id_map(raw)


def merge_gold_maps(gold_main, gold_round2):
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



def polarity_upper(label: str):
    p = label_to_polarity(label)
    if p == 'pos':
        return 'POS'
    if p == 'neg':
        return 'NEG'
    return None


def polarity_set(labels):
    out = set()
    for lb in labels:
        p = polarity_upper(lb)
        if p is not None:
            out.add(p)
    return out


def grad_set_between(labels_a, labels_b) -> Set[int]:
    pa_set = polarity_set(labels_a)
    pb_set = polarity_set(labels_b)
    if not pa_set or not pb_set:
        return set()
    out = set()
    for pa in pa_set:
        for pb in pb_set:
            if pa == pb:
                out.add(0)
            elif pa == 'NEG' and pb == 'POS':
                out.add(+1)
            elif pa == 'POS' and pb == 'NEG':
                out.add(-1)
    return out


def grad_match(g_a, g_b, p_a, p_b):
    gg = grad_set_between(g_a, g_b)
    pp = grad_set_between(p_a, p_b)
    valid = (len(gg) > 0 and len(pp) > 0)
    ok = 1 if (valid and len(gg & pp) > 0) else 0
    return valid, ok


def context_augmented_improved_score_pred_driven(gold_map, pred2, pred3):
    total = 0
    correct = 0
    common_pred_ids = sorted(set(pred2.keys()) & set(pred3.keys()))
    for iid in common_pred_ids:
        if iid not in gold_map:
            continue
        g = gold_map[iid]
        g2 = to_list_labels(extract_stage_label(g, 'stage2'), normalize_text)
        g3 = to_list_labels(extract_stage_label(g, 'stage3'), normalize_text)
        if not g2 or not g3:
            continue
        p2 = to_list_labels(extract_stage_label(pred2[iid], 'stage2'), normalize_text)
        p3 = to_list_labels(extract_stage_label(pred3[iid], 'stage3'), normalize_text)
        total += 1
        valid, ok = grad_match(g2, g3, p2, p3)
        if valid and ok:
            correct += 1
    return total, (correct / total if total else 0.0)


def main():
    parser = build_common_parser('9.5 Stage2→3 Gradient Consistency Rate')
    args = parser.parse_args()
    io = resolve_common_args(args)
    check_required_files([io.gold_base, io.gold_round2, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gold_map = merge_gold_maps(
        load_gold_valid_records(io.gold_base, load_json_or_jsonl),
        load_gold_valid_records(io.gold_round2, load_json_or_jsonl),
    )
    pred2 = load_pred_stage(io.pred_stage2, 'stage2', load_json_or_jsonl)
    pred3 = load_pred_stage(io.pred_stage3, 'stage3', load_json_or_jsonl)
    denom, score = context_augmented_improved_score_pred_driven(gold_map, pred2, pred3)
    print(f'stage2_to_3_gradient_consistency_rate: {score:.4f} | 分母={denom}')


if __name__ == '__main__':
    main()
