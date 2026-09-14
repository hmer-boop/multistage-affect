# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

from metric_runtime_common_cot import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    load_json_or_jsonl,
    rows_from_obj,
    print_metric_io,
    merge_pred_maps,
)

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)


def _as_list(x) -> List[str]:
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
        iid = r.get('item_id') or r.get('id')
        if not iid:
            continue
        iid = str(iid)
        gold[iid] = {
            'stage1': _as_list(r.get('stage1')),
            'stage2': _as_list(r.get('stage2')),
            'stage3': _as_list(r.get('stage3')),
        }
    return gold


def merge_gold(base_map, extra_map):
    merged = dict(base_map)
    for iid, info in extra_map.items():
        if iid in merged:
            continue
        merged[iid] = info
    return merged


def _first_label(raw) -> str:
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, str):
        return raw.strip()
    return ''


def _extract_pred_label(record: dict, stage: str):
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


def load_pred_stage(path, stage: str) -> Dict[str, str]:
    obj = load_json_or_jsonl(path)
    rows = rows_from_obj(obj, path)
    pred = {}
    for r in rows:
        iid = r.get('id') or r.get('item_id')
        if not iid:
            continue
        lab = _extract_pred_label(r, stage)
        pred[str(iid)] = _first_label(lab)
    return pred


def is_correct(pred_label: str, gold_labels: List[str]) -> bool:
    return pred_label in LABEL_SET and pred_label in gold_labels


def eval_stage_pred_driven(gold, pred, stage: str) -> Tuple[int, int, float]:
    total = 0
    correct = 0
    for iid, pred_label in pred.items():
        if iid not in gold:
            continue
        gold_labels = gold[iid].get(stage, [])
        if not gold_labels:
            continue
        total += 1
        if is_correct(pred_label, gold_labels):
            correct += 1
    return total, correct, (correct / total if total else 0.0)


def eval_tsmr_pred_driven(gold, pred1, pred2, pred3) -> Tuple[int, float]:
    common_pred_ids = sorted(set(pred1.keys()) & set(pred2.keys()) & set(pred3.keys()))
    valid_ids = []
    for iid in common_pred_ids:
        if iid not in gold:
            continue
        g = gold[iid]
        if g.get('stage1') and g.get('stage2') and g.get('stage3'):
            valid_ids.append(iid)
    if not valid_ids:
        return 0, 0.0
    tsmr_sum = 0.0
    for iid in valid_ids:
        g = gold[iid]
        c1 = is_correct(pred1.get(iid, ''), g['stage1'])
        c2 = is_correct(pred2.get(iid, ''), g['stage2'])
        c3 = is_correct(pred3.get(iid, ''), g['stage3'])
        tsmr_sum += (int(c1) + int(c2) + int(c3)) / 3.0
    return len(valid_ids), tsmr_sum / len(valid_ids)


def main():
    parser = build_common_parser('9.1 Stage Label Accuracy + TSMR')
    args = parser.parse_args()
    io = resolve_common_args(args)
    print_metric_io(io, quiet=getattr(args, "quiet_paths", False))
    required = [io.gold_base, io.gold_round2, io.pred_stage1, io.pred_stage2, io.pred_stage3]
    required += [p for p in [io.pred_stage1_round2, io.pred_stage2_round2, io.pred_stage3_round2] if p]
    check_required_files(required)

    gold = merge_gold(load_gold_single(io.gold_base), load_gold_single(io.gold_round2))
    pred1 = merge_pred_maps(
        load_pred_stage(io.pred_stage1, 'stage1'),
        load_pred_stage(io.pred_stage1_round2, 'stage1') if io.pred_stage1_round2 else {},
    )
    pred2 = merge_pred_maps(
        load_pred_stage(io.pred_stage2, 'stage2'),
        load_pred_stage(io.pred_stage2_round2, 'stage2') if io.pred_stage2_round2 else {},
    )
    pred3 = merge_pred_maps(
        load_pred_stage(io.pred_stage3, 'stage3'),
        load_pred_stage(io.pred_stage3_round2, 'stage3') if io.pred_stage3_round2 else {},
    )

    s1_total, _, s1_acc = eval_stage_pred_driven(gold, pred1, 'stage1')
    s2_total, _, s2_acc = eval_stage_pred_driven(gold, pred2, 'stage2')
    s3_total, _, s3_acc = eval_stage_pred_driven(gold, pred3, 'stage3')
    tsmr_total, tsmr = eval_tsmr_pred_driven(gold, pred1, pred2, pred3)

    print(f'stage1: {s1_acc:.4f} | 分母={s1_total}')
    print(f'stage2: {s2_acc:.4f} | 分母={s2_total}')
    print(f'stage3: {s3_acc:.4f} | 分母={s3_total}')
    print(f'TSMR: {tsmr:.4f} | 分母={tsmr_total}')


if __name__ == '__main__':
    main()
