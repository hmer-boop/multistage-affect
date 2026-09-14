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
- stage2_to_3_gradient_consistency_rate：仅 6/7/8 号 Stage3 消融补充计算 2→3 极性梯度一致率（9.5）
- context_supplement_gain_rate：仅 6/7/8 号 Stage3 消融补充计算 Stage2错→Stage3对的语境补充增益率（9.6）
- context_regression_rate / CRR：仅 6/7/8 号 Stage3 消融补充计算 Stage2对→Stage3错的语境回退率

说明：
- 当前脚本只评价“单个消融 setting 的单阶段结果”。
- 不计算 TSMR。
- 不计算 9.2b 三阶段极性一致率。
- Stage3 各设置统一使用 ArtTIDE Stage2 预测作为参照，计算 9.5、9.6 和 CRR。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =========================================================
# 0. 路径配置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLD_BASE = PROJECT_ROOT / "results" / "gold_label.json"
GOLD_ROUND2 = PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json"

ABLATION_ROOT = PROJECT_ROOT / "000gpt4o_ablation"

EVAL_OUT_DIR = PROJECT_ROOT / "9.3_evaluation_ablation" / "results"

# Stage3 转移指标统一使用的 Stage2 参照。该文件也是 Stage3
# 消融输入中 image_prior_label 的来源。
ARTTIDE_STAGE2_PRED = (
    PROJECT_ROOT / "01improvement method" / "method data" / "stage2_gpt_labels4.json"
)


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
        "pred_stage2": ARTTIDE_STAGE2_PRED,
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
        "pred_stage2": ARTTIDE_STAGE2_PRED,
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
        "pred_stage2": ARTTIDE_STAGE2_PRED,
    },
    "9": {
        "name": "stage3_base_common_stage2_reference",
        "stage": "stage3",
        "setting": "base",
        "pred_path": PROJECT_ROOT / "result gpt_stage3" / "results_stage3.json",
        "pred_path_round2": (
            PROJECT_ROOT / "result gpt_stage3" / "results_stage3_round2.json"
        ),
        "pred_stage2": ARTTIDE_STAGE2_PRED,
    },
    "10": {
        "name": "stage3_arttide_full_common_stage2_reference",
        "stage": "stage3",
        "setting": "arttide_full",
        "pred_path": (
            PROJECT_ROOT / "01improvement method" / "method data" / "stage3_gpt_labels3.json"
        ),
        "pred_stage2": ARTTIDE_STAGE2_PRED,
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


def merge_pred_maps(pred_base: Dict[str, Dict[str, Any]], pred_round2: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    合并 base 与 round2 预测结果。

    说明：
    - 用于将分开保存的两批预测合并后评价；
    - 若出现重复 id，默认保留 base 中的记录，避免 round2 意外覆盖原始预测。
    """
    merged = dict(pred_base)
    for item_id, rec in pred_round2.items():
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
# 9. Stage2 → Stage3 Gradient Consistency Rate（9.5）
# =========================================================

def polarity_upper(label: str) -> Optional[str]:
    """
    将情感标签转为 POS / NEG。
    与 9.5 COT 脚本保持一致：
    - 正向：宁静、快乐、惊奇、敬畏
    - 负向：悲伤、恐惧、厌恶、愤怒
    """
    p = label_to_polarity(label)
    if p == "pos":
        return "POS"
    if p == "neg":
        return "NEG"
    return None


def polarity_set(labels: List[str]) -> Set[str]:
    out: Set[str] = set()
    for lb in labels:
        p = polarity_upper(lb)
        if p is not None:
            out.add(p)
    return out


def grad_set_between(labels_a: List[str], labels_b: List[str]) -> Set[int]:
    """
    计算两个阶段之间的极性变化集合：
    -  0：极性不变
    - +1：NEG → POS
    - -1：POS → NEG

    如果任一阶段标签无法映射到极性，则返回空集合。
    """
    pa_set = polarity_set(labels_a)
    pb_set = polarity_set(labels_b)

    if not pa_set or not pb_set:
        return set()

    out: Set[int] = set()
    for pa in pa_set:
        for pb in pb_set:
            if pa == pb:
                out.add(0)
            elif pa == "NEG" and pb == "POS":
                out.add(+1)
            elif pa == "POS" and pb == "NEG":
                out.add(-1)
    return out


def grad_match(gold_stage2: List[str], gold_stage3: List[str], pred_stage2: List[str], pred_stage3: List[str]) -> Tuple[bool, int]:
    """
    判断预测的 Stage2→Stage3 极性变化是否命中金标变化。

    注意：
    - valid=False 表示 gold 或 pred 有一侧无法形成有效极性梯度；
    - 但在 pred-driven 分母中，预测无效不会被剔除，而是计入分母并算错。
    """
    gold_grad = grad_set_between(gold_stage2, gold_stage3)
    pred_grad = grad_set_between(pred_stage2, pred_stage3)

    valid = (len(gold_grad) > 0 and len(pred_grad) > 0)
    ok = 1 if (valid and len(gold_grad & pred_grad) > 0) else 0
    return valid, ok


def eval_stage2_to_3_gradient_consistency(
    gold_map: Dict[str, Dict[str, Any]],
    pred_stage2_map: Dict[str, Dict[str, Any]],
    pred_stage3_map: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, float]:
    """
    9.5：Stage2→Stage3 极性梯度一致率。

    分母策略与 COT 版 9.5 保持一致：
    - 以 pred_stage2 与 pred_stage3 的共同 id 为出发点；
    - id 必须存在于 gold；
    - gold stage2 与 gold stage3 均非空；
    - pred 标签非法/空白时仍进入分母，并作为错误处理。
    """
    total = 0
    correct = 0

    common_pred_ids = sorted(set(pred_stage2_map.keys()) & set(pred_stage3_map.keys()))

    for item_id in common_pred_ids:
        if item_id not in gold_map:
            continue

        gold_rec = gold_map[item_id]
        g2 = to_list_labels(extract_stage_label(gold_rec, "stage2"))
        g3 = to_list_labels(extract_stage_label(gold_rec, "stage3"))

        if not g2 or not g3:
            continue

        p2 = to_list_labels(extract_stage_label(pred_stage2_map[item_id], "stage2"))
        p3 = to_list_labels(extract_stage_label(pred_stage3_map[item_id], "stage3"))

        total += 1

        valid, ok = grad_match(g2, g3, p2, p3)
        if valid and ok:
            correct += 1

    score = correct / total if total else 0.0
    return total, correct, score


# =========================================================
# 10. Context Supplement Gain Rate（9.6）
# =========================================================

def label_hit(pred_labels: List[str], gold_labels: List[str]) -> bool:
    """
    阶段标签命中规则与 Label Accuracy 保持一致：
    - 金标允许单标签或双标签；
    - 预测标签命中任一金标标签即算正确；
    - 预测标签不在 LABEL_SET 中时按错误处理。
    """
    if not pred_labels or not gold_labels:
        return False
    return any((lb in LABEL_SET and lb in gold_labels) for lb in pred_labels)


def eval_context_supplement_gain_rate(
    gold_map: Dict[str, Dict[str, Any]],
    pred_stage2_map: Dict[str, Dict[str, Any]],
    pred_stage3_map: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, int, float, int, int, int, float]:
    """
    9.6：语境补充增益率 Context Supplement Gain Rate。

    计算口径：
        CSGR = #(Stage2 错误 且 Stage3 正确) / #(Stage2 错误)

    消融评估中的数据来源：
    - Stage2：固定读取 ArtTIDE 的 stage2 预测文件；
    - Stage3：读取当前 stage3 消融 setting 的预测文件；
    - gold：读取 gold_label.json + gold_label_round2.json。

    分母策略：
    - 以 ArtTIDE stage2 与当前 stage3 setting 的共同 id 为出发点；
    - id 必须存在于 gold；
    - gold stage2 与 gold stage3 均非空；
    - pred 标签非法/空白时仍进入判断，并按错误处理。
    """
    eval_total = 0
    stage2_correct_count = 0
    stage3_correct_count = 0
    stage2_wrong_count = 0
    context_gain_count = 0
    context_regression_count = 0

    common_pred_ids = sorted(set(pred_stage2_map.keys()) & set(pred_stage3_map.keys()))

    for item_id in common_pred_ids:
        if item_id not in gold_map:
            continue

        gold_rec = gold_map[item_id]
        g2 = to_list_labels(extract_stage_label(gold_rec, "stage2"))
        g3 = to_list_labels(extract_stage_label(gold_rec, "stage3"))

        if not g2 or not g3:
            continue

        p2 = to_list_labels(extract_stage_label(pred_stage2_map[item_id], "stage2"))
        p3 = to_list_labels(extract_stage_label(pred_stage3_map[item_id], "stage3"))

        s2_correct = label_hit(p2, g2)
        s3_correct = label_hit(p3, g3)

        eval_total += 1

        if s2_correct:
            stage2_correct_count += 1
        else:
            stage2_wrong_count += 1

        if s3_correct:
            stage3_correct_count += 1

        if (not s2_correct) and s3_correct:
            context_gain_count += 1

        if s2_correct and (not s3_correct):
            context_regression_count += 1

    gain_rate = context_gain_count / stage2_wrong_count if stage2_wrong_count else 0.0
    regression_rate = context_regression_count / stage2_correct_count if stage2_correct_count else 0.0

    return (
        eval_total,
        stage2_wrong_count,
        context_gain_count,
        gain_rate,
        stage2_correct_count,
        stage3_correct_count,
        context_regression_count,
        regression_rate,
    )


# =========================================================
# 11. 单个 setting 评价
# =========================================================

def eval_one_setting(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stage = cfg["stage"]
    setting = cfg["setting"]
    pred_path: Path = cfg["pred_path"]
    pred_path_round2_raw = cfg.get("pred_path_round2")
    pred_path_round2 = Path(pred_path_round2_raw) if pred_path_round2_raw else None

    print("\n========================================")
    print(f"Evaluate Ablation Setting: {cfg['name']}")
    print("========================================")
    print(f"- stage: {stage}")
    print(f"- setting: {setting}")
    print(f"- pred_path: {pred_path}")
    if pred_path_round2 is not None:
        print(f"- pred_path_round2: {pred_path_round2}")
    if stage == "stage3":
        print(f"- Stage2 reference: {cfg.get('pred_stage2', ARTTIDE_STAGE2_PRED)}")
    print(f"- gold_base: {GOLD_BASE}")
    print(f"- gold_round2: {GOLD_ROUND2}")
    print("========================================")

    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在：{pred_path}")
    if pred_path_round2 is not None and not pred_path_round2.exists():
        raise FileNotFoundError(f"round2 预测文件不存在：{pred_path_round2}")

    gold_map = merge_gold_maps(
        load_gold_map(GOLD_BASE),
        load_gold_map(GOLD_ROUND2),
    )
    pred_map = load_pred_map(pred_path, stage)
    if pred_path_round2 is not None:
        pred_map = merge_pred_maps(
            pred_map,
            load_pred_map(pred_path_round2, stage),
        )

    label_n, label_correct, label_acc = eval_label_accuracy(gold_map, pred_map, stage)
    pol_n, pol_correct, pol_acc = eval_polarity_accuracy(gold_map, pred_map, stage)
    pol_micro_n, pol_mp, pol_mr, pol_mf1 = calc_polarity_micro(gold_map, pred_map, stage)
    ml_n, ml_mp, ml_mr, ml_mf1 = calc_multilabel_micro(gold_map, pred_map, stage)

    grad_n = grad_correct = 0
    grad_acc: Optional[float] = None

    context_eval_n = 0
    context_stage2_wrong_n = 0
    context_gain_correct = 0
    context_gain_rate: Optional[float] = None
    context_stage2_correct_n = 0
    context_stage3_correct_n = 0
    context_regression_n = 0
    context_regression_rate: Optional[float] = None

    pred_stage2_path: Optional[Path] = None
    if stage == "stage3":
        pred_stage2_path = Path(cfg.get("pred_stage2", ARTTIDE_STAGE2_PRED))

        if not pred_stage2_path.exists():
            raise FileNotFoundError(f"Stage2 reference 预测文件不存在：{pred_stage2_path}")

        pred_stage2_map = load_pred_map(pred_stage2_path, "stage2")

        grad_n, grad_correct, grad_acc = eval_stage2_to_3_gradient_consistency(
            gold_map=gold_map,
            pred_stage2_map=pred_stage2_map,
            pred_stage3_map=pred_map,
        )

        (
            context_eval_n,
            context_stage2_wrong_n,
            context_gain_correct,
            context_gain_rate,
            context_stage2_correct_n,
            context_stage3_correct_n,
            context_regression_n,
            context_regression_rate,
        ) = eval_context_supplement_gain_rate(
            gold_map=gold_map,
            pred_stage2_map=pred_stage2_map,
            pred_stage3_map=pred_map,
        )

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

    if stage == "stage3":
        result.update({
            "stage2_to_3_gradient_consistency_rate": round(grad_acc or 0.0, 4),
            "stage2_to_3_gradient_correct": grad_correct,
            "stage2_to_3_gradient_N": grad_n,

            "context_supplement_gain_rate": round(context_gain_rate or 0.0, 4),
            "context_supplement_gain_correct": context_gain_correct,
            "context_supplement_gain_N": context_stage2_wrong_n,
            "context_eval_N": context_eval_n,
            "context_stage2_correct_N": context_stage2_correct_n,
            "context_stage3_correct_N": context_stage3_correct_n,
            "context_regression_count": context_regression_n,
            "context_regression_rate": round(context_regression_rate or 0.0, 4),
            "CRR": round(context_regression_rate or 0.0, 4),
            "CRR_count": context_regression_n,
            "CRR_N": context_stage2_correct_n,

            "stage2_reference_pred_path": str(pred_stage2_path),
            "stage3_ablation_pred_path": str(pred_path),
        })

    print("\n---------- 指标结果 ----------")
    print(f"{stage_short}L: {label_acc:.4f} | correct={label_correct} | 分母={label_n}")
    print(f"{stage_short}P: {pol_acc:.4f} | correct={pol_correct} | 分母={pol_n}")
    print(f"Polarity MP:  {pol_mp:.4f} | 分母={pol_micro_n}")
    print(f"Polarity MR:  {pol_mr:.4f} | 分母={pol_micro_n}")
    print(f"Polarity MF1: {pol_mf1:.4f} | 分母={pol_micro_n}")
    print(f"Multilabel MP:  {ml_mp:.4f} | 分母={ml_n}")
    print(f"Multilabel MR:  {ml_mr:.4f} | 分母={ml_n}")
    print(f"Multilabel MF1: {ml_mf1:.4f} | 分母={ml_n}")

    if stage == "stage3":
        print(f"Stage2→3 Gradient Consistency Rate: {(grad_acc or 0.0):.4f} | correct={grad_correct} | 分母={grad_n}")
        print(f"Context Supplement Gain Rate: {(context_gain_rate or 0.0):.4f} | gain={context_gain_correct} | 分母(Stage2错误)={context_stage2_wrong_n}")
        print(f"  - Context eval N: {context_eval_n} | Stage2 correct={context_stage2_correct_n} | Stage3 correct={context_stage3_correct_n}")
        print(f"Context Regression Rate (CRR): {(context_regression_rate or 0.0):.4f} | regression={context_regression_n} | 分母(Stage2正确)={context_stage2_correct_n}")
        print(f"  - Stage2对→Stage3错: {context_regression_n} | rate={(context_regression_rate or 0.0):.4f}")
        print(f"  - 使用 Stage2 reference: {pred_stage2_path}")
        print(f"  - 使用 Stage3 ablation: {pred_path}")

    out_path = EVAL_OUT_DIR / f"{cfg['name']}_metrics.json"
    save_json(out_path, result)
    print(f"\n已保存指标结果：{out_path}")

    return result


# =========================================================
# 12. 交互入口
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
