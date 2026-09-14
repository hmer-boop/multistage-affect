# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from metric_runtime_common_cot import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    load_json_or_jsonl,
    normalize_text,
    print_metric_io,
    merge_pred_maps,
)

STAGES = ['stage1', 'stage2', 'stage3']

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

    preferred = [
        stage,
        f'{stage}_label',
        f'{stage}_labels',
        f'{stage}_pred',
        f'{stage}_prediction',
        f'{stage}_pred_label',
        f'{stage}_emotion',
        f'{stage}_emotion_label',
        f'results_{stage}',
    ]
    for key in preferred:
        if key in record:
            return record[key]

    # 防止把 reason / thought / cot 字段误当作标签，只接受明显像标签/预测结果的字段。
    for k, v in record.items():
        kl = str(k).lower()
        if stage in kl and any(tok in kl for tok in ['label', 'pred', 'emotion', 'result', 'answer']):
            return v

    generic = [
        'label', 'pred_label', 'prediction_label', 'emotion_label', 'emotion',
        'pred', 'prediction', 'answer', 'final_label', 'result', 'output',
    ]
    for key in generic:
        if key in record:
            return record[key]
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


def counts_to_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def calc_polarity_counts_for_stage(gold_map, pred_map, stage: str) -> Dict[str, int]:
    """
    单阶段标准 polarity micro 统计。
    - 以 pred_map 为出发点；
    - gold 缺失或该阶段 gold 为空则跳过；
    - 在 pos/neg 两个维度上分别累计 TP/FP/FN；
    - pred 为空/非法时保留 gold 产生的 FN，并额外计 1 个 FP。
    """
    tp = fp = fn = 0
    denom = 0

    for item_id, prec in pred_map.items():
        if item_id not in gold_map:
            continue

        g_labels = to_list_labels(extract_stage_label(gold_map[item_id], stage), normalize_text)
        if not g_labels:
            continue

        p_labels = to_list_labels(extract_stage_label(prec, stage), normalize_text)
        g_pols = labels_to_polarity_set(g_labels)
        p_pols = labels_to_polarity_set(p_labels)

        if not g_pols:
            continue

        denom += 1

        for polarity in ('pos', 'neg'):
            gold_positive = polarity in g_pols
            pred_positive = polarity in p_pols
            if gold_positive and pred_positive:
                tp += 1
            elif not gold_positive and pred_positive:
                fp += 1
            elif gold_positive and not pred_positive:
                fn += 1

        if not p_pols:
            fp += 1

    return {'denom': denom, 'tp': tp, 'fp': fp, 'fn': fn}


def calc_micro_f1_pred_driven(gold_map, pred1, pred2, pred3):
    stage_maps = {'stage1': pred1, 'stage2': pred2, 'stage3': pred3}
    stage_results: Dict[str, Dict[str, float]] = {}

    total_denom = total_tp = total_fp = total_fn = 0
    for stage in STAGES:
        counts = calc_polarity_counts_for_stage(gold_map, stage_maps[stage], stage)
        p, r, f1 = counts_to_metrics(counts['tp'], counts['fp'], counts['fn'])
        stage_results[stage] = {
            'denom': counts['denom'],
            'precision': p,
            'recall': r,
            'f1': f1,
        }
        total_denom += counts['denom']
        total_tp += counts['tp']
        total_fp += counts['fp']
        total_fn += counts['fn']

    overall_p, overall_r, overall_f1 = counts_to_metrics(total_tp, total_fp, total_fn)
    return total_denom, overall_p, overall_r, overall_f1, stage_results


def main():
    parser = build_common_parser('9.3 Three-stage Polarity Accuracy')
    args = parser.parse_args()
    io = resolve_common_args(args)
    print_metric_io(io, quiet=getattr(args, "quiet_paths", False))
    required = [io.gold_base, io.gold_round2, io.pred_stage1, io.pred_stage2, io.pred_stage3]
    required += [p for p in [io.pred_stage1_round2, io.pred_stage2_round2, io.pred_stage3_round2] if p]
    check_required_files(required)

    gold_map = merge_gold_maps(
        load_gold_valid_records(io.gold_base, load_json_or_jsonl),
        load_gold_valid_records(io.gold_round2, load_json_or_jsonl),
    )
    pred1 = merge_pred_maps(
        load_pred_stage(io.pred_stage1, 'stage1', load_json_or_jsonl),
        load_pred_stage(io.pred_stage1_round2, 'stage1', load_json_or_jsonl) if io.pred_stage1_round2 else {},
    )
    pred2 = merge_pred_maps(
        load_pred_stage(io.pred_stage2, 'stage2', load_json_or_jsonl),
        load_pred_stage(io.pred_stage2_round2, 'stage2', load_json_or_jsonl) if io.pred_stage2_round2 else {},
    )
    pred3 = merge_pred_maps(
        load_pred_stage(io.pred_stage3, 'stage3', load_json_or_jsonl),
        load_pred_stage(io.pred_stage3_round2, 'stage3', load_json_or_jsonl) if io.pred_stage3_round2 else {},
    )

    denom, p, r, f1, stage_results = calc_micro_f1_pred_driven(gold_map, pred1, pred2, pred3)

    print('===== 9.3 3-Stage Polarity Accuracy =====')
    print(f'参与样本数: {denom}')
    print(f'Micro Precision: {p:.4f} | 分母={denom}')
    print(f'Micro Recall: {r:.4f} | 分母={denom}')
    print(f'Micro F1: {f1:.4f} | 分母={denom}')

    for stage in STAGES:
        sr = stage_results[stage]
        sd = int(sr['denom'])
        print(f'{stage} Precision: {sr["precision"]:.4f} | 分母={sd}')
        print(f'{stage} Recall: {sr["recall"]:.4f} | 分母={sd}')
        print(f'{stage} F1: {sr["f1"]:.4f} | 分母={sd}')


if __name__ == '__main__':
    main()
