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

POS = {"宁静", "快乐", "惊奇", "敬畏"}
NEG = {"悲伤", "恐惧", "厌恶", "愤怒"}
STAGES = ["stage1", "stage2", "stage3"]


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if "\n" in text and text.lstrip().startswith("{") and text.count("\n{") >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def label_to_polarity(label):
    if label in POS:
        return "pos"
    if label in NEG:
        return "neg"
    return None


def labels_to_polarity(labels):
    return {label_to_polarity(l) for l in labels if label_to_polarity(l)}


def to_id_map(data):
    out = {}
    if isinstance(data, list):
        for x in data:
            if not isinstance(x, dict):
                continue
            iid = x.get("item_id") or x.get("id")
            if iid is None:
                continue
            out[str(iid)] = x
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv["item_id"] = k
                out[str(k)] = vv
    return out


def merge_pred_maps(*maps):
    pred_map = {}
    for src in maps:
        for iid, rec in src.items():
            pred_map.setdefault(iid, {}).update(rec)
    return pred_map


def get_valid_eval_ids(gold_map, pred_map, mode: str):
    candidate_ids = choose_candidate_ids(gold_map, pred_map, mode)
    valid_ids = []
    for iid in candidate_ids:
        grec = gold_map[iid]
        ok = True
        for st in STAGES:
            g_labels = parse_valid_label_list(extract_stage_value(grec, st))
            if not g_labels:
                ok = False
                break
        if ok:
            valid_ids.append(iid)
    return valid_ids


def main():
    parser = build_common_parser("9.3 Three-stage Polarity Accuracy")
    args = parser.parse_args()
    io = resolve_common_args(args)

    check_required_files([io.gold_base, io.pred_stage1, io.pred_stage2, io.pred_stage3])

    gold_map = {}
    gold_map.update(to_id_map(load_json_or_jsonl(io.gold_base)))
    if io.gold_round2 and io.gold_round2.exists():
        gold_map.update(to_id_map(load_json_or_jsonl(io.gold_round2)))

    pred_map = merge_pred_maps(
        to_id_map(load_json_or_jsonl(io.pred_stage1)) if io.pred_stage1 and io.pred_stage1.exists() else {},
        to_id_map(load_json_or_jsonl(io.pred_stage2)) if io.pred_stage2 and io.pred_stage2.exists() else {},
        to_id_map(load_json_or_jsonl(io.pred_stage3)) if io.pred_stage3 and io.pred_stage3.exists() else {},
        to_id_map(load_json_or_jsonl(io.pred_stage1_round2)) if io.pred_stage1_round2 and io.pred_stage1_round2.exists() else {},
        to_id_map(load_json_or_jsonl(io.pred_stage2_round2)) if io.pred_stage2_round2 and io.pred_stage2_round2.exists() else {},
        to_id_map(load_json_or_jsonl(io.pred_stage3_round2)) if io.pred_stage3_round2 and io.pred_stage3_round2.exists() else {},
    )

    valid_ids = get_valid_eval_ids(gold_map, pred_map, io.eval_id_mode)

    TP = FP = FN = 0
    stage_stats = {stage: {"TP": 0, "FP": 0, "FN": 0, "n": 0} for stage in STAGES}

    for item_id in valid_ids:
        grec = gold_map[item_id]
        prec = pred_map[item_id]
        for st in STAGES:
            g_labels = parse_valid_label_list(extract_stage_value(grec, st)) or []
            p_labels = parse_valid_label_list(extract_stage_value(prec, st)) or []
            g_pols = labels_to_polarity(g_labels)
            p_pols = labels_to_polarity(p_labels)
            if not g_pols:
                continue

            stage_stats[st]["n"] += 1

            for polarity in ("pos", "neg"):
                gold_positive = polarity in g_pols
                pred_positive = polarity in p_pols
                if gold_positive and pred_positive:
                    TP += 1
                    stage_stats[st]["TP"] += 1
                elif not gold_positive and pred_positive:
                    FP += 1
                    stage_stats[st]["FP"] += 1
                elif gold_positive and not pred_positive:
                    FN += 1
                    stage_stats[st]["FN"] += 1

            # Invalid outputs are outside the polarity label space. Count one
            # additional FP so they are penalized as failed predictions.
            if not p_pols:
                FP += 1
                stage_stats[st]["FP"] += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def stage_prf(s):
        p = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) > 0 else 0.0
        r = s["TP"] / (s["TP"] + s["FN"]) if (s["TP"] + s["FN"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    s1p, s1r, s1f = stage_prf(stage_stats["stage1"])
    s2p, s2r, s2f = stage_prf(stage_stats["stage2"])
    s3p, s3r, s3f = stage_prf(stage_stats["stage3"])

    print("===== 9.3 3-Stage Polarity Accuracy =====")
    print(f"参与样本数: {len(valid_ids)}")
    print(f"Micro Precision: {precision:.4f}")
    print(f"Micro Recall: {recall:.4f}")
    print(f"Micro F1: {f1:.4f}")
    print(f"stage1 Precision: {s1p:.4f}")
    print(f"stage1 Recall: {s1r:.4f}")
    print(f"stage1 F1: {s1f:.4f}")
    print(f"stage2 Precision: {s2p:.4f}")
    print(f"stage2 Recall: {s2r:.4f}")
    print(f"stage2 F1: {s2f:.4f}")
    print(f"stage3 Precision: {s3p:.4f}")
    print(f"stage3 Recall: {s3r:.4f}")
    print(f"stage3 F1: {s3f:.4f}")


if __name__ == "__main__":
    main()
