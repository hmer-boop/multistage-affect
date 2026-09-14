# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from metric_runtime_common_improve import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    load_json_or_jsonl,
    normalize_text,
)

STAGES = ['stage1', 'stage2', 'stage3']
LABELS = ['宁静', '快乐', '惊奇', '敬畏', '悲伤', '恐惧', '厌恶', '愤怒']
LABEL_TO_IDX = {x: i for i, x in enumerate(LABELS)}
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


def labels_to_multihot(labels: List[str]):
    vec = [0] * len(LABELS)
    for lb in labels:
        if lb in LABEL_TO_IDX:
            vec[LABEL_TO_IDX[lb]] = 1
    return vec


def calc_micro_metrics_multilabel(y_true, y_pred) -> Tuple[int, float, float, float]:
    tp = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        for a, b in zip(yt, yp):
            if a == 1 and b == 1:
                tp += 1
            elif a == 0 and b == 1:
                fp += 1
            elif a == 1 and b == 0:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return len(y_true), precision, recall, f1


def collect_stage_multilabel_vectors(gold_map, pred_map, stage: str):
    y_true, y_pred = [], []
    for item_id, prec in pred_map.items():
        if item_id not in gold_map:
            continue

        gold_labels = to_list_labels(extract_stage_label(gold_map[item_id], stage), normalize_text)
        if not gold_labels:
            continue

        pred_labels = to_list_labels(extract_stage_label(prec, stage), normalize_text)
        y_true.append(labels_to_multihot(gold_labels))
        y_pred.append(labels_to_multihot(pred_labels))

    return y_true, y_pred


def main():
    parser = build_common_parser('9.4 Three-stage Multi-label Accuracy')
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

    pred_maps = {'stage1': pred1, 'stage2': pred2, 'stage3': pred3}

    overall_y_true, overall_y_pred = [], []
    stage_results: Dict[str, Dict[str, float]] = {}

    for stage in STAGES:
        y_true, y_pred = collect_stage_multilabel_vectors(gold_map, pred_maps[stage], stage)
        overall_y_true.extend(y_true)
        overall_y_pred.extend(y_pred)

        denom_s, p_s, r_s, f1_s = calc_micro_metrics_multilabel(y_true, y_pred)
        stage_results[stage] = {
            'denom': denom_s,
            'precision': p_s,
            'recall': r_s,
            'f1': f1_s,
        }

    denom, p, r, f1 = calc_micro_metrics_multilabel(overall_y_true, overall_y_pred)

    print('===== 9.4 3-Stage Multi-label Accuracy =====')
    print(f'参与样本数: {denom}')
    print(f'Overall Precision: {p:.4f} | 分母={denom}')
    print(f'Overall Recall: {r:.4f} | 分母={denom}')
    print(f'Overall F1: {f1:.4f} | 分母={denom}')

    for stage in STAGES:
        sr = stage_results[stage]
        sd = int(sr['denom'])
        print(f'{stage} Precision: {sr["precision"]:.4f} | 分母={sd}')
        print(f'{stage} Recall: {sr["recall"]:.4f} | 分母={sd}')
        print(f'{stage} F1: {sr["f1"]:.4f} | 分母={sd}')


if __name__ == '__main__':
    main()
