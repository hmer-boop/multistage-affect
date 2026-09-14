# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

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



def evaluate_pred_driven(gold_map, pred_map, stage: str) -> Tuple[int, int, float]:
    total = 0
    correct = 0
    for item_id, prec in pred_map.items():
        if item_id not in gold_map:
            continue
        g_labels = to_list_labels(extract_stage_label(gold_map[item_id], stage), normalize_text)
        if not g_labels:
            continue
        p_labels = to_list_labels(extract_stage_label(prec, stage), normalize_text)
        gold_pols = labels_to_polarity_set(g_labels)
        pred_pols = labels_to_polarity_set(p_labels)
        total += 1
        if gold_pols and pred_pols and (gold_pols & pred_pols):
            correct += 1
    return total, correct, (correct / total if total else 0.0)


def main():
    parser = build_common_parser('9.2a One-stage Polarity Accuracy')
    args = parser.parse_args()
    io = resolve_common_args(args)
    check_required_files([io.gold_base, io.gold_round2, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gold_map = merge_gold_maps(
        load_gold_valid_records(io.gold_base, load_json_or_jsonl),
        load_gold_valid_records(io.gold_round2, load_json_or_jsonl),
    )
    pred1 = load_pred_stage(io.pred_stage1, 'stage1', load_json_or_jsonl)
    pred2 = load_pred_stage(io.pred_stage2, 'stage2', load_json_or_jsonl)
    pred3 = load_pred_stage(io.pred_stage3, 'stage3', load_json_or_jsonl)

    s1_total, _, s1_acc = evaluate_pred_driven(gold_map, pred1, 'stage1')
    s2_total, _, s2_acc = evaluate_pred_driven(gold_map, pred2, 'stage2')
    s3_total, _, s3_acc = evaluate_pred_driven(gold_map, pred3, 'stage3')

    print(f'stage1_polarity: {s1_acc:.4f} | 分母={s1_total}')
    print(f'stage2_polarity: {s2_acc:.4f} | 分母={s2_total}')
    print(f'stage3_polarity: {s3_acc:.4f} | 分母={s3_total}')


if __name__ == '__main__':
    main()
