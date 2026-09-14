# -*- coding: utf-8 -*-
"""
eval_stage_ablation.py

消融实验专用评价脚本。

当前支持：
1) Stage1 - s1_wo_tonal
2) Stage1 - s1_wo_composition

输出指标：
- S1L / S2L / S3L：当前阶段标签准确率
- S1P / S2P / S3P：当前阶段极性准确率
- Polarity MP / MR / MF1：当前阶段极性 micro precision / recall / F1
- Multilabel MP / MR / MF1：当前阶段多标签 micro precision / recall / F1

说明：
- 当前脚本只评价“单个消融 setting 的单阶段结果”。
- 不计算 TSMR。
- 不计算 9.2b 三阶段极性一致率。
- Stage3 的 2→3 指标之后单独加。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# 0. 路径配置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLD_BASE = PROJECT_ROOT / "results" / "gold_label.json"
GOLD_ROUND2 = PROJECT_ROOT / "results" / "gold_label_round2.json"

ABLATION_ROOT = PROJECT_ROOT / "000gpt4o_ablation"

EVAL_OUT_DIR = PROJECT_ROOT / "9.3_evaluation_ablation" / "results"


# =========================================================
# 1. 消融实验注册表
# =========================================================

EVAL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # =====================================================
    # Stage 1 ablation
    # =====================================================
    "1": {
        "name": "stage1_s1_wo_tonal",
        "stage": "stage1",
        "setting": "s1_wo_tonal",
        "pred_path": ABLATION_ROOT
        / "outputs"
        / "stage1"
        / "s1_wo_tonal"
        / "s1_wo_tonal_stage1_predictions.json",
    },
    "2": {
        "name": "stage1_s1_wo_composition",
        "stage": "stage1",
        "setting": "s1_wo_composition",
        "pred_path": ABLATION_ROOT
        / "outputs"
        / "stage1"
        / "s1_wo_composition"
        / "s1_wo_composition_stage1_predictions.json",
    },
    "3": {
        "name": "stage1_s1_wo_dynamism",
        "stage": "stage1",
        "setting": "s1_wo_dynamism",
        "pred_path": ABLATION_ROOT
        / "outputs"
        / "stage1"
        / "s1_wo_dynamism"
        / "s1_wo_dynamism_stage1_predictions.json",
    },

    # =====================================================
    # Stage 2 ablation
    # =====================================================
    "4": {
        "name": "stage2_s2_entity_only",
        "stage": "stage2",
        "setting": "s2_entity_only",
        "pred_path": ABLATION_ROOT
        / "outputs"
        / "stage2"
        / "s2_entity_only"
        / "s2_entity_only_stage2_predictions.json",
    },
    "5": {
        "name": "stage2_s2_entity_relation",
        "stage": "stage2",
        "setting": "s2_entity_relation",
        "pred_path": ABLATION_ROOT
        / "outputs"
        / "stage2"
        / "s2_entity_relation"
        / "s2_entity_relation_stage2_predictions.json",
    },

    # =====================================================
    # Stage 3 ablation
    # =====================================================
    "6": {
        "name": "stage3_s3_wo_image_prior",
        "stage": "stage3",
        "setting": "s3_wo_image_prior",
        "pred_path": ABLATION_ROOT
        / "outputs"
        / "stage3"
        / "s3_wo_image_prior"
        / "s3_wo_image_prior_stage3_predictions.json",
    },
    "7": {
        "name": "stage3_s3_wo_textual_structured_prior",
        "stage": "stage3",
        "setting": "s3_wo_textual_structured_prior",
        "pred_path": ABLATION_ROOT
        / "outputs"
        / "stage3"
        / "s3_wo_textual_structured_prior"
        / "s3_wo_textual_structured_prior_stage3_predictions.json",
    },
    "8": {
        "name": "stage3_s3_wo_textual_explanation",
        "stage": "stage3",
        "setting": "s3_wo_textual_explanation",
        "pred_path": ABLATION_ROOT
        / "outputs"
        / "stage3"
        / "s3_wo_textual_explanation"
        / "s3_wo_textual_explanation_stage3_predictions.json",
    },
}

# =========================================================
# 2. 标签与极性配置
# =========================================================

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)

# multilabel 顺序保持和你原 9.4 接近
MULTI_LABELS = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]
LABEL_TO_IDX = {x: i for i, x in enumerate(MULTI_LABELS)}

POS_SET = {
    "宁静", "平静", "安宁", "恬静",
    "快乐", "喜悦", "高兴", "愉悦", "幸福",
    "惊奇", "惊讶",
    "敬畏", "崇敬", "肃然起敬",
}
NEG_SET = {
    "悲伤", "伤感", "忧伤",
    "恐惧", "害怕", "畏惧",
    "厌恶", "反感", "恶心",
    "愤怒", "生气", "恼怒", "气愤",
}
CANON_POS = {"宁静", "快乐", "惊奇", "敬畏"}
CANON_NEG = {"悲伤", "恐惧", "厌恶", "愤怒"}

SPLITTERS = ["/", "、", "|", ",", "，", ";", "；"]


# =========================================================
# 3. 基础工具函数
# =========================================================

def load_json_or_jsonl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    # 简单兼容 jsonl
    if "\n" in text and text.lstrip().startswith("{") and text.count("\n{") >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    return json.loads(text)


def rows_from_obj(obj: Any) -> List[dict]:
    if isinstance(obj, dict) and isinstance(obj.get("valid"), list):
        rows = obj["valid"]
    elif isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = list(obj.values())
    else:
        raise ValueError("JSON 格式不符合预期：应为 list、dict 或包含 valid 的 dict")

    return [r for r in rows if isinstance(r, dict)]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize_text(x: Any) -> str:
    return str(x).strip().strip("“”\"' 　\t")


def to_list_labels(x: Any) -> List[str]:
    """
    gold 可能是 list，也可能是 string。
    pred 通常是 string。
    """
    if x is None:
        return []

    if isinstance(x, dict):
        if "labels" in x:
            return to_list_labels(x.get("labels"))
        return []

    if isinstance(x, list):
        out: List[str] = []
        for v in x:
            out.extend(to_list_labels(v))
        return [normalize_text(v) for v in out if normalize_text(v)]

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
    """
    兼容多种字段：
    - stage1_label
    - stage1
    - prediction
    - label
    """
    if not record:
        return None

    preferred = [
        f"{stage}_label",
        stage,
        "prediction",
        "label",
        "pred",
    ]

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
                v2["item_id"] = str(v.get("item_id") or v.get("id") or k)
                id_map[str(v2["item_id"])] = v2
        return id_map

    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue
            item_id = rec.get("item_id") or rec.get("id")
            if item_id is None:
                continue
            id_map[str(item_id)] = rec

    return id_map


# =========================================================
# 4. 读取 gold 与 pred
# =========================================================

def load_gold_map(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = load_json_or_jsonl(path)
    rows = rows_from_obj(raw)
    return records_to_id_map(rows)


def merge_gold_maps(gold_base: Dict[str, Dict[str, Any]], gold_round2: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged = dict(gold_base)
    for item_id, rec in gold_round2.items():
        if item_id in merged:
            continue
        merged[item_id] = rec
    return merged


def load_pred_map(path: Path, stage: str) -> Dict[str, Dict[str, Any]]:
    raw = load_json_or_jsonl(path)
    rows = rows_from_obj(raw)
    id_map = records_to_id_map(rows)

    pred_map: Dict[str, Dict[str, Any]] = {}
    for item_id, rec in id_map.items():
        pred_map[item_id] = {
            stage: extract_stage_label(rec, stage)
        }

    return pred_map


# =========================================================
# 5. Label Accuracy
# =========================================================

def eval_label_accuracy(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], stage: str) -> Tuple[int, int, float]:
    """
    分母：
    - 以 pred 文件中的 id 为出发点；
    - 该 id 在 gold 中存在；
    - 当前 stage 的 gold 非空。

    pred 非法 / 空白：
    - 不剔除出分母，直接算错。
    """
    total = 0
    correct = 0

    for item_id, pred_rec in pred_map.items():
        if item_id not in gold_map:
            continue

        gold_labels = to_list_labels(extract_stage_label(gold_map[item_id], stage))
        if not gold_labels:
            continue

        pred_labels = to_list_labels(extract_stage_label(pred_rec, stage))
        pred_label = pred_labels[0] if pred_labels else ""

        total += 1

        if pred_label in LABEL_SET and pred_label in gold_labels:
            correct += 1

    acc = correct / total if total else 0.0
    return total, correct, acc


# =========================================================
# 6. Polarity Accuracy
# =========================================================

def label_to_polarity(label: str) -> Optional[str]:
    if not label:
        return None
    if label in POS_SET or label in CANON_POS:
        return "pos"
    if label in NEG_SET or label in CANON_NEG:
        return "neg"
    return None


def labels_to_polarity_set(labels: List[str]) -> set:
    return {p for p in (label_to_polarity(lb) for lb in labels) if p}


def eval_polarity_accuracy(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], stage: str) -> Tuple[int, int, float]:
    total = 0
    correct = 0

    for item_id, pred_rec in pred_map.items():
        if item_id not in gold_map:
            continue

        gold_labels = to_list_labels(extract_stage_label(gold_map[item_id], stage))
        if not gold_labels:
            continue

        pred_labels = to_list_labels(extract_stage_label(pred_rec, stage))

        gold_pols = labels_to_polarity_set(gold_labels)
        pred_pols = labels_to_polarity_set(pred_labels)

        total += 1

        if gold_pols and pred_pols and (gold_pols & pred_pols):
            correct += 1

    acc = correct / total if total else 0.0
    return total, correct, acc


# =========================================================
# 7. Polarity Micro P / R / F1
# =========================================================

def calc_polarity_micro(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], stage: str) -> Tuple[int, float, float, float]:
    tp = fp = fn = 0
    compared = 0

    for item_id, pred_rec in pred_map.items():
        if item_id not in gold_map:
            continue

        gold_labels = to_list_labels(extract_stage_label(gold_map[item_id], stage))
        if not gold_labels:
            continue

        pred_labels = to_list_labels(extract_stage_label(pred_rec, stage))

        gold_pols = labels_to_polarity_set(gold_labels)
        pred_pols = labels_to_polarity_set(pred_labels)

        compared += 1

        if not gold_pols:
            continue

        for polarity in ("pos", "neg"):
            gold_positive = polarity in gold_pols
            pred_positive = polarity in pred_pols
            if gold_positive and pred_positive:
                tp += 1
            elif not gold_positive and pred_positive:
                fp += 1
            elif gold_positive and not pred_positive:
                fn += 1

        if not pred_pols:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return compared, precision, recall, f1


# =========================================================
# 8. Multilabel Micro P / R / F1
# =========================================================

def labels_to_multihot(labels: List[str]) -> List[int]:
    vec = [0] * len(MULTI_LABELS)
    for lb in labels:
        if lb in LABEL_TO_IDX:
            vec[LABEL_TO_IDX[lb]] = 1
    return vec


def calc_multilabel_micro(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], stage: str) -> Tuple[int, float, float, float]:
    y_true: List[List[int]] = []
    y_pred: List[List[int]] = []

    for item_id, pred_rec in pred_map.items():
        if item_id not in gold_map:
            continue

        gold_labels = to_list_labels(extract_stage_label(gold_map[item_id], stage))
        if not gold_labels:
            continue

        pred_labels = to_list_labels(extract_stage_label(pred_rec, stage))

        y_true.append(labels_to_multihot(gold_labels))
        y_pred.append(labels_to_multihot(pred_labels))

    tp = fp = fn = 0

    for yt, yp in zip(y_true, y_pred):
        for a, b in zip(yt, yp):
            if a == 1 and b == 1:
                tp += 1
            elif a == 0 and b == 1:
                fp += 1
            elif a == 1 and b == 0:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return len(y_true), precision, recall, f1


# =========================================================
# 9. 单个 setting 评价
# =========================================================

def eval_one_setting(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stage = cfg["stage"]
    setting = cfg["setting"]
    pred_path: Path = cfg["pred_path"]

    print("\n========================================")
    print(f"Evaluate Ablation Setting: {cfg['name']}")
    print("========================================")
    print(f"- stage: {stage}")
    print(f"- setting: {setting}")
    print(f"- pred_path: {pred_path}")
    print(f"- gold_base: {GOLD_BASE}")
    print(f"- gold_round2: {GOLD_ROUND2}")
    print("========================================")

    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在：{pred_path}")

    gold_map = merge_gold_maps(
        load_gold_map(GOLD_BASE),
        load_gold_map(GOLD_ROUND2),
    )
    pred_map = load_pred_map(pred_path, stage)

    label_n, label_correct, label_acc = eval_label_accuracy(gold_map, pred_map, stage)
    pol_n, pol_correct, pol_acc = eval_polarity_accuracy(gold_map, pred_map, stage)
    pol_micro_n, pol_mp, pol_mr, pol_mf1 = calc_polarity_micro(gold_map, pred_map, stage)
    ml_n, ml_mp, ml_mr, ml_mf1 = calc_multilabel_micro(gold_map, pred_map, stage)

    stage_short = {
        "stage1": "S1",
        "stage2": "S2",
        "stage3": "S3",
    }.get(stage, stage)

    result = {
        "name": cfg["name"],
        "stage": stage,
        "setting": setting,
        "pred_path": str(pred_path),

        f"{stage_short}L": round(label_acc, 4),
        f"{stage_short}L_correct": label_correct,
        f"{stage_short}L_N": label_n,

        f"{stage_short}P": round(pol_acc, 4),
        f"{stage_short}P_correct": pol_correct,
        f"{stage_short}P_N": pol_n,

        "polarity_MP": round(pol_mp, 4),
        "polarity_MR": round(pol_mr, 4),
        "polarity_MF1": round(pol_mf1, 4),
        "polarity_micro_N": pol_micro_n,

        "multilabel_MP": round(ml_mp, 4),
        "multilabel_MR": round(ml_mr, 4),
        "multilabel_MF1": round(ml_mf1, 4),
        "multilabel_micro_N": ml_n,
    }

    print("\n---------- 指标结果 ----------")
    print(f"{stage_short}L: {label_acc:.4f} | correct={label_correct} | 分母={label_n}")
    print(f"{stage_short}P: {pol_acc:.4f} | correct={pol_correct} | 分母={pol_n}")
    print(f"Polarity MP:  {pol_mp:.4f} | 分母={pol_micro_n}")
    print(f"Polarity MR:  {pol_mr:.4f} | 分母={pol_micro_n}")
    print(f"Polarity MF1: {pol_mf1:.4f} | 分母={pol_micro_n}")
    print(f"Multilabel MP:  {ml_mp:.4f} | 分母={ml_n}")
    print(f"Multilabel MR:  {ml_mr:.4f} | 分母={ml_n}")
    print(f"Multilabel MF1: {ml_mf1:.4f} | 分母={ml_n}")

    out_path = EVAL_OUT_DIR / f"{cfg['name']}_metrics.json"
    save_json(out_path, result)
    print(f"\n已保存指标结果：{out_path}")

    return result


# =========================================================
# 10. 交互入口
# =========================================================

def list_settings() -> None:
    print("\n===== 可评价的消融实验 =====")
    for key in sorted(EVAL_REGISTRY.keys(), key=lambda x: int(x)):
        cfg = EVAL_REGISTRY[key]
        print(f"{key}. {cfg['name']} | {cfg['stage']} | {cfg['setting']}")
    print()


def main() -> None:
    list_settings()
    choice = input("请输入要评价的消融实验编号：").strip()

    if choice not in EVAL_REGISTRY:
        raise ValueError(f"编号不存在：{choice}")

    cfg = EVAL_REGISTRY[choice]
    eval_one_setting(cfg)


if __name__ == "__main__":
    main()
