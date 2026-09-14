from __future__ import annotations

import json
from pathlib import Path

from metric_runtime_common import (
    build_common_parser,
    resolve_common_args,
    check_required_files,
    parse_valid_label_list,
    extract_stage_value,
    choose_candidate_ids,
)

STAGES = ["stage1", "stage2", "stage3"]
LABELS = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]
LABEL_TO_IDX = {x: i for i, x in enumerate(LABELS)}


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if "\n" in text and text.lstrip().startswith("{") and text.count("\n{") >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def to_item_map(rows):
    mp = {}
    if isinstance(rows, list):
        for x in rows:
            if not isinstance(x, dict):
                continue
            iid = x.get("item_id") or x.get("id")
            if iid is not None:
                mp[str(iid)] = x
    elif isinstance(rows, dict):
        for k, v in rows.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv["item_id"] = k
                mp[str(k)] = vv
    return mp


def merge_stage_maps(*maps):
    merged = {}
    for src in maps:
        for iid, rec in src.items():
            merged.setdefault(iid, {}).update(rec)
    return merged


def labels_to_multihot(labels):
    vec = [0] * len(LABELS)
    for lb in labels:
        if lb in LABEL_TO_IDX:
            vec[LABEL_TO_IDX[lb]] = 1
    return vec


def calc_micro_metrics_multilabel(y_true, y_pred):
    if not y_true or not y_pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "sample_count": 0, "tp": 0, "fp": 0, "fn": 0}
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
    return {"precision": precision, "recall": recall, "f1": f1, "sample_count": len(y_true), "tp": tp, "fp": fp, "fn": fn}


def get_valid_eval_ids(gold_map, pred_map, mode: str):
    candidate_ids = choose_candidate_ids(gold_map, pred_map, mode)
    valid_ids = []
    for iid in candidate_ids:
        gold_item = gold_map[iid]
        pred_item = pred_map[iid]
        ok = True
        for stage in STAGES:
            gold_labels = parse_valid_label_list(extract_stage_value(gold_item, stage))
            pred_labels = parse_valid_label_list(extract_stage_value(pred_item, stage))
            if not gold_labels or not pred_labels:
                ok = False
                break
        if ok:
            valid_ids.append(iid)
    return valid_ids


def main():
    parser = build_common_parser("9.4 Three-stage Multi-label Accuracy")
    args = parser.parse_args()
    io = resolve_common_args(args)

    check_required_files([io.gold_base, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gpt_map = merge_stage_maps(
        to_item_map(load_json_or_jsonl(io.pred_stage1)) if io.pred_stage1 and io.pred_stage1.exists() else {},
        to_item_map(load_json_or_jsonl(io.pred_stage2)) if io.pred_stage2 and io.pred_stage2.exists() else {},
        to_item_map(load_json_or_jsonl(io.pred_stage3)) if io.pred_stage3 and io.pred_stage3.exists() else {},
        to_item_map(load_json_or_jsonl(io.pred_stage1_round2)) if io.pred_stage1_round2 and io.pred_stage1_round2.exists() else {},
        to_item_map(load_json_or_jsonl(io.pred_stage2_round2)) if io.pred_stage2_round2 and io.pred_stage2_round2.exists() else {},
        to_item_map(load_json_or_jsonl(io.pred_stage3_round2)) if io.pred_stage3_round2 and io.pred_stage3_round2.exists() else {},
    )

    gold_map = {}
    gold_map.update(to_item_map(load_json_or_jsonl(io.gold_base)))
    if io.gold_round2 and io.gold_round2.exists():
        gold_map.update(to_item_map(load_json_or_jsonl(io.gold_round2)))

    valid_ids = get_valid_eval_ids(gold_map, gpt_map, io.eval_id_mode)

    overall_y_true, overall_y_pred = [], []
    stage_y_true = {stage: [] for stage in STAGES}
    stage_y_pred = {stage: [] for stage in STAGES}

    for item_id in valid_ids:
        gold_item = gold_map[item_id]
        gpt_item = gpt_map[item_id]
        for stage in STAGES:
            gpt_labels = parse_valid_label_list(extract_stage_value(gpt_item, stage)) or []
            gold_labels = parse_valid_label_list(extract_stage_value(gold_item, stage)) or []
            true_vec = labels_to_multihot(gold_labels)
            pred_vec = labels_to_multihot(gpt_labels)
            overall_y_true.append(true_vec)
            overall_y_pred.append(pred_vec)
            stage_y_true[stage].append(true_vec)
            stage_y_pred[stage].append(pred_vec)

    overall_metrics = calc_micro_metrics_multilabel(overall_y_true, overall_y_pred)
    stage_metrics = {stage: calc_micro_metrics_multilabel(stage_y_true[stage], stage_y_pred[stage]) for stage in STAGES}

    print("===== 9.4 3-Stage Multi-Label Accuracy =====")
    print(f"参与样本数: {len(valid_ids)}")
    print(f"Overall Precision: {overall_metrics['precision']:.4f}")
    print(f"Overall Recall: {overall_metrics['recall']:.4f}")
    print(f"Overall F1: {overall_metrics['f1']:.4f}")
    print(f"stage1 Precision: {stage_metrics['stage1']['precision']:.4f}")
    print(f"stage1 Recall: {stage_metrics['stage1']['recall']:.4f}")
    print(f"stage1 F1: {stage_metrics['stage1']['f1']:.4f}")
    print(f"stage2 Precision: {stage_metrics['stage2']['precision']:.4f}")
    print(f"stage2 Recall: {stage_metrics['stage2']['recall']:.4f}")
    print(f"stage2 F1: {stage_metrics['stage2']['f1']:.4f}")
    print(f"stage3 Precision: {stage_metrics['stage3']['precision']:.4f}")
    print(f"stage3 Recall: {stage_metrics['stage3']['recall']:.4f}")
    print(f"stage3 F1: {stage_metrics['stage3']['f1']:.4f}")


if __name__ == "__main__":
    main()
