from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_REGISTRY_PATH = PROJECT_ROOT / "9_evaluation" / "run_eval_center.py"
ARTTIDE_REGISTRY_PATH = PROJECT_ROOT / "9.2_evaluation_improve" / "run_eval_center_improve.py"
EVAL_GOLD_ROUND2 = PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json"

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)
MULTILABEL_ORDER = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]
MULTILABEL_INDEX = {label: i for i, label in enumerate(MULTILABEL_ORDER)}
STAGES = ("stage1", "stage2", "stage3")
SPLITTERS = ["/", "、", "|", ",", "，", ";", "；"]


MODEL_PAIRS = [
    ("GPT-4o", 1, 1),
    ("Gemini-2.5-Pro", 2, 2),
    ("InternVL-8B", 3, 3),
    ("LLaVA-4B", 7, 4),
    ("Qwen-VL-8B", 5, 5),
]


@dataclass
class PredictionBundle:
    gold: Dict[str, Dict[str, Any]]
    pred: Dict[str, Dict[str, Any]]


@dataclass
class MetricResult:
    model: str
    metric: str
    direction: str
    n_items: int
    base_value: float
    arttide_value: float
    improvement: float
    ci_low: float
    ci_high: float
    excludes_zero: bool


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import registry from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if "\n" in text and text.lstrip().startswith("{") and text.count("\n{") >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def rows_from_obj(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("valid"), list):
        rows = obj["valid"]
    elif isinstance(obj, dict) and isinstance(obj.get("results"), list):
        rows = obj["results"]
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        rows = obj["data"]
    elif isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = list(obj.values())
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def normalize_text(x: Any) -> str:
    return str(x).strip().strip("“”\"' 　\t")


def split_labels(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        if "labels" in x:
            return split_labels(x.get("labels"))
        for key in ("label", "pred_label", "prediction", "answer", "final_label", "result", "output"):
            if key in x:
                return split_labels(x.get(key))
        return []
    if isinstance(x, list):
        out: List[str] = []
        for value in x:
            out.extend(split_labels(value))
        return [normalize_text(value) for value in out if normalize_text(value)]
    value = normalize_text(x)
    if not value:
        return []
    parts = [value]
    for splitter in SPLITTERS:
        new_parts: List[str] = []
        for part in parts:
            new_parts.extend(part.split(splitter))
        parts = new_parts
    return [normalize_text(value) for value in parts if normalize_text(value)]


def parse_valid_label_list(x: Any) -> Optional[List[str]]:
    labels = split_labels(x)
    if not labels:
        return None
    return labels if all(label in LABEL_SET for label in labels) else None


def as_gold_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[str] = []
        for value in x:
            if value is not None:
                text = normalize_text(value)
                if text:
                    out.append(text)
        return out
    text = normalize_text(x)
    return [text] if text else []


def first_pred_label(x: Any) -> str:
    if isinstance(x, list):
        x = x[0] if x else None
    if isinstance(x, str):
        return x.strip()
    return ""


def extract_stage_value(record: Dict[str, Any], stage: str) -> Any:
    preferred = [
        stage,
        f"{stage}_label",
        f"{stage}_labels",
        f"{stage}_pred",
        f"{stage}_prediction",
        f"{stage}_pred_label",
        f"{stage}_emotion",
        f"{stage}_emotion_label",
        f"results_{stage}",
    ]
    for key in preferred:
        if key in record:
            return record[key]
    for key, value in record.items():
        key_lower = str(key).lower()
        if stage in key_lower and any(token in key_lower for token in ["label", "pred", "emotion", "result", "answer"]):
            return value
    for key in ("label", "pred_label", "prediction_label", "emotion_label", "emotion", "pred", "prediction", "answer", "final_label", "result", "output"):
        if key in record:
            return record[key]
    return None


def to_item_map(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = rows_from_obj(load_json_or_jsonl(path))
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id") or row.get("id")
        if item_id is not None:
            out[str(item_id)] = row
    return out


def load_gold_map(gold_base: Path, gold_round2: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    merged = dict(to_item_map(gold_base))
    if gold_round2 and gold_round2.exists():
        for item_id, rec in to_item_map(gold_round2).items():
            if item_id not in merged:
                merged[item_id] = rec
    return merged


def merge_pred_maps(*maps: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for mp in maps:
        merged.update(mp)
    return merged


def load_stage_pred(path: Optional[Path], stage: str) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Any] = {}
    for item_id, rec in to_item_map(path).items():
        out[item_id] = extract_stage_value(rec, stage)
    return out


def load_base_bundle(cfg: Dict[str, Any]) -> PredictionBundle:
    gold = load_gold_map(PROJECT_ROOT / "results" / "gold_label.json", EVAL_GOLD_ROUND2)
    pred = {
        stage: merge_pred_maps(
            load_stage_pred(Path(cfg[f"pred_{stage}"]), stage),
            load_stage_pred(Path(cfg.get(f"pred_{stage}_round2", "")), stage) if cfg.get(f"pred_{stage}_round2") else {},
        )
        for stage in STAGES
    }
    return PredictionBundle(gold=gold, pred=pred)


def load_arttide_bundle(cfg: Dict[str, Any]) -> PredictionBundle:
    gold = load_gold_map(Path(cfg["gold_base"]), Path(cfg["gold_round2"]))
    pred = {stage: load_stage_pred(Path(cfg[f"pred_{stage}"]), stage) for stage in STAGES}
    return PredictionBundle(gold=gold, pred=pred)


def gold_labels(bundle: PredictionBundle, item_id: str, stage: str) -> List[str]:
    return as_gold_list(extract_stage_value(bundle.gold[item_id], stage))


def first_hit(bundle: PredictionBundle, item_id: str, stage: str) -> bool:
    pred_label = first_pred_label(bundle.pred[stage].get(item_id))
    return pred_label in LABEL_SET and pred_label in gold_labels(bundle, item_id, stage)


def parsed_hit(bundle: PredictionBundle, item_id: str, stage: str) -> bool:
    pred_labels = parse_valid_label_list(bundle.pred[stage].get(item_id))
    gold = parse_valid_label_list(extract_stage_value(bundle.gold[item_id], stage))
    return bool(pred_labels and gold and (set(pred_labels) & set(gold)))


def valid_gold_ids(bundle: PredictionBundle, stages: Iterable[str]) -> List[str]:
    ids = []
    for item_id, rec in bundle.gold.items():
        ok = True
        for stage in stages:
            if not as_gold_list(extract_stage_value(rec, stage)):
                ok = False
                break
        if ok:
            ids.append(item_id)
    return ids


def common_stage_ids(base: PredictionBundle, art: PredictionBundle, stage: str) -> List[str]:
    ids = set(valid_gold_ids(base, [stage]))
    ids &= set(base.pred[stage].keys())
    ids &= set(art.pred[stage].keys())
    return sorted(ids)


def common_trajectory_ids(base: PredictionBundle, art: PredictionBundle) -> List[str]:
    ids = set(valid_gold_ids(base, STAGES))
    for stage in STAGES:
        ids &= set(base.pred[stage].keys())
        ids &= set(art.pred[stage].keys())
    return sorted(ids)


def common_context_ids(base: PredictionBundle, art: PredictionBundle) -> List[str]:
    ids = set()
    for item_id in valid_gold_ids(base, ("stage2", "stage3")):
        if (
            item_id in base.pred["stage2"]
            and item_id in base.pred["stage3"]
            and item_id in art.pred["stage2"]
            and item_id in art.pred["stage3"]
        ):
            ids.add(item_id)
    return sorted(ids)


def stage_arrays(base: PredictionBundle, art: PredictionBundle, stage: str) -> tuple[List[str], np.ndarray, np.ndarray]:
    ids = common_stage_ids(base, art, stage)
    base_arr = np.array([1.0 if first_hit(base, item_id, stage) else 0.0 for item_id in ids], dtype=np.float64)
    art_arr = np.array([1.0 if first_hit(art, item_id, stage) else 0.0 for item_id in ids], dtype=np.float64)
    return ids, base_arr, art_arr


def tsmr_arrays(base: PredictionBundle, art: PredictionBundle) -> tuple[List[str], np.ndarray, np.ndarray]:
    ids = common_trajectory_ids(base, art)
    base_arr = np.array([
        sum(1.0 if first_hit(base, item_id, stage) else 0.0 for stage in STAGES) / 3.0
        for item_id in ids
    ], dtype=np.float64)
    art_arr = np.array([
        sum(1.0 if first_hit(art, item_id, stage) else 0.0 for stage in STAGES) / 3.0
        for item_id in ids
    ], dtype=np.float64)
    return ids, base_arr, art_arr


def sta_arrays(base: PredictionBundle, art: PredictionBundle) -> tuple[List[str], np.ndarray, np.ndarray]:
    ids = common_trajectory_ids(base, art)
    base_arr = np.array([
        1.0 if all(first_hit(base, item_id, stage) for stage in STAGES) else 0.0
        for item_id in ids
    ], dtype=np.float64)
    art_arr = np.array([
        1.0 if all(first_hit(art, item_id, stage) for stage in STAGES) else 0.0
        for item_id in ids
    ], dtype=np.float64)
    return ids, base_arr, art_arr


def context_components(bundle: PredictionBundle, ids: Sequence[str]) -> Dict[str, np.ndarray]:
    s2_correct = []
    s2_wrong = []
    gain = []
    regression = []
    for item_id in ids:
        c2 = parsed_hit(bundle, item_id, "stage2")
        c3 = parsed_hit(bundle, item_id, "stage3")
        s2_correct.append(1.0 if c2 else 0.0)
        s2_wrong.append(0.0 if c2 else 1.0)
        gain.append(1.0 if (not c2 and c3) else 0.0)
        regression.append(1.0 if (c2 and not c3) else 0.0)
    return {
        "s2_correct": np.array(s2_correct, dtype=np.float64),
        "s2_wrong": np.array(s2_wrong, dtype=np.float64),
        "gain": np.array(gain, dtype=np.float64),
        "regression": np.array(regression, dtype=np.float64),
    }


def context_value(components: Dict[str, np.ndarray], numerator_key: str, denominator_key: str, idx: Optional[np.ndarray] = None) -> np.ndarray | float:
    numerator = components[numerator_key]
    denominator = components[denominator_key]
    if idx is None:
        den = float(denominator.sum())
        return float(numerator.sum() / den) if den else 0.0
    num = numerator[idx].sum(axis=1)
    den = denominator[idx].sum(axis=1)
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return out


def label_multihot(labels: Optional[List[str]]) -> List[int]:
    vec = [0] * len(MULTILABEL_ORDER)
    for label in labels or []:
        if label in MULTILABEL_INDEX:
            vec[MULTILABEL_INDEX[label]] = 1
    return vec


def multilabel_item_counts(bundle: PredictionBundle, ids: Sequence[str]) -> Dict[str, np.ndarray]:
    tp, fp, fn = [], [], []
    for item_id in ids:
        item_tp = item_fp = item_fn = 0
        for stage in STAGES:
            true_vec = label_multihot(parse_valid_label_list(extract_stage_value(bundle.gold[item_id], stage)))
            pred_vec = label_multihot(parse_valid_label_list(bundle.pred[stage].get(item_id)))
            for true_value, pred_value in zip(true_vec, pred_vec):
                if true_value == 1 and pred_value == 1:
                    item_tp += 1
                elif true_value == 0 and pred_value == 1:
                    item_fp += 1
                elif true_value == 1 and pred_value == 0:
                    item_fn += 1
        tp.append(item_tp)
        fp.append(item_fp)
        fn.append(item_fn)
    return {
        "tp": np.array(tp, dtype=np.float64),
        "fp": np.array(fp, dtype=np.float64),
        "fn": np.array(fn, dtype=np.float64),
    }


def micro_f1(counts: Dict[str, np.ndarray], idx: Optional[np.ndarray] = None) -> np.ndarray | float:
    if idx is None:
        tp = float(counts["tp"].sum())
        fp = float(counts["fp"].sum())
        fn = float(counts["fn"].sum())
    else:
        tp = counts["tp"][idx].sum(axis=1)
        fp = counts["fp"][idx].sum(axis=1)
        fn = counts["fn"][idx].sum(axis=1)
    precision_den = tp + fp
    recall_den = tp + fn
    if idx is None:
        precision = tp / precision_den if precision_den else 0.0
        recall = tp / recall_den if recall_den else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    precision = np.zeros_like(tp, dtype=np.float64)
    recall = np.zeros_like(tp, dtype=np.float64)
    np.divide(tp, precision_den, out=precision, where=precision_den != 0)
    np.divide(tp, recall_den, out=recall, where=recall_den != 0)
    denom = precision + recall
    out = np.zeros_like(tp, dtype=np.float64)
    np.divide(2 * precision * recall, denom, out=out, where=denom != 0)
    return out


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="linear"))


def bootstrap_mean_diff(
    base_arr: np.ndarray,
    art_arr: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    reverse: bool = False,
) -> tuple[float, float, float, float, float]:
    n = len(base_arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    base_value = float(base_arr.mean())
    art_value = float(art_arr.mean())
    improvement = (base_value - art_value) if reverse else (art_value - base_value)
    idx = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int32)
    base_boot = base_arr[idx].mean(axis=1)
    art_boot = art_arr[idx].mean(axis=1)
    boot_diff = (base_boot - art_boot) if reverse else (art_boot - base_boot)
    return base_value, art_value, improvement, percentile(boot_diff, 2.5), percentile(boot_diff, 97.5)


def bootstrap_ratio_diff(
    base_components: Dict[str, np.ndarray],
    art_components: Dict[str, np.ndarray],
    numerator_key: str,
    denominator_key: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    reverse: bool = False,
) -> tuple[float, float, float, float, float]:
    n = len(base_components[numerator_key])
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    base_value = float(context_value(base_components, numerator_key, denominator_key))
    art_value = float(context_value(art_components, numerator_key, denominator_key))
    improvement = (base_value - art_value) if reverse else (art_value - base_value)
    idx = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int32)
    base_boot = context_value(base_components, numerator_key, denominator_key, idx)
    art_boot = context_value(art_components, numerator_key, denominator_key, idx)
    boot_diff = (base_boot - art_boot) if reverse else (art_boot - base_boot)
    return base_value, art_value, improvement, percentile(boot_diff, 2.5), percentile(boot_diff, 97.5)


def bootstrap_f1_diff(
    base_counts: Dict[str, np.ndarray],
    art_counts: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float, float, float]:
    n = len(base_counts["tp"])
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    base_value = float(micro_f1(base_counts))
    art_value = float(micro_f1(art_counts))
    improvement = art_value - base_value
    idx = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int32)
    boot_diff = micro_f1(art_counts, idx) - micro_f1(base_counts, idx)
    return base_value, art_value, improvement, percentile(boot_diff, 2.5), percentile(boot_diff, 97.5)


def build_result(
    model: str,
    metric: str,
    direction: str,
    n_items: int,
    values: tuple[float, float, float, float, float],
) -> MetricResult:
    base_value, art_value, improvement, ci_low, ci_high = values
    return MetricResult(
        model=model,
        metric=metric,
        direction=direction,
        n_items=n_items,
        base_value=base_value,
        arttide_value=art_value,
        improvement=improvement,
        ci_low=ci_low,
        ci_high=ci_high,
        excludes_zero=(ci_low > 0.0 or ci_high < 0.0),
    )


def evaluate_model(model_name: str, base: PredictionBundle, art: PredictionBundle, rng: np.random.Generator, n_bootstrap: int) -> List[MetricResult]:
    results: List[MetricResult] = []

    for stage, metric_name in (("stage1", "S1L"), ("stage2", "S2L"), ("stage3", "S3L")):
        ids, base_arr, art_arr = stage_arrays(base, art, stage)
        results.append(build_result(
            model_name,
            metric_name,
            "ArtTIDE-Base",
            len(ids),
            bootstrap_mean_diff(base_arr, art_arr, rng, n_bootstrap),
        ))

    ids, base_arr, art_arr = tsmr_arrays(base, art)
    results.append(build_result(
        model_name,
        "TSMR",
        "ArtTIDE-Base",
        len(ids),
        bootstrap_mean_diff(base_arr, art_arr, rng, n_bootstrap),
    ))

    ids, base_arr, art_arr = sta_arrays(base, art)
    results.append(build_result(
        model_name,
        "STA",
        "ArtTIDE-Base",
        len(ids),
        bootstrap_mean_diff(base_arr, art_arr, rng, n_bootstrap),
    ))

    context_ids = common_context_ids(base, art)
    base_context = context_components(base, context_ids)
    art_context = context_components(art, context_ids)
    results.append(build_result(
        model_name,
        "CSGR",
        "ArtTIDE-Base",
        len(context_ids),
        bootstrap_ratio_diff(base_context, art_context, "gain", "s2_wrong", rng, n_bootstrap),
    ))
    results.append(build_result(
        model_name,
        "CRR",
        "Base-ArtTIDE",
        len(context_ids),
        bootstrap_ratio_diff(base_context, art_context, "regression", "s2_correct", rng, n_bootstrap, reverse=True),
    ))

    ml_ids = common_trajectory_ids(base, art)
    base_counts = multilabel_item_counts(base, ml_ids)
    art_counts = multilabel_item_counts(art, ml_ids)
    results.append(build_result(
        model_name,
        "ML-MF1",
        "ArtTIDE-Base",
        len(ml_ids),
        bootstrap_f1_diff(base_counts, art_counts, rng, n_bootstrap),
    ))

    return results


def write_csv(path: Path, results: Sequence[MetricResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "metric",
                "direction",
                "n_items",
                "base_value",
                "arttide_value",
                "improvement",
                "improvement_pp",
                "ci_low",
                "ci_high",
                "ci_low_pp",
                "ci_high_pp",
                "excludes_zero",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                "model": result.model,
                "metric": result.metric,
                "direction": result.direction,
                "n_items": result.n_items,
                "base_value": f"{result.base_value:.8f}",
                "arttide_value": f"{result.arttide_value:.8f}",
                "improvement": f"{result.improvement:.8f}",
                "improvement_pp": f"{100 * result.improvement:.4f}",
                "ci_low": f"{result.ci_low:.8f}",
                "ci_high": f"{result.ci_high:.8f}",
                "ci_low_pp": f"{100 * result.ci_low:.4f}",
                "ci_high_pp": f"{100 * result.ci_high:.4f}",
                "excludes_zero": "yes" if result.excludes_zero else "no",
            })


def write_json(path: Path, results: Sequence[MetricResult], n_bootstrap: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "unit": "artwork item",
        "interval": "percentile 95% confidence interval",
        "results": [
            {
                "model": r.model,
                "metric": r.metric,
                "direction": r.direction,
                "n_items": r.n_items,
                "base_value": r.base_value,
                "arttide_value": r.arttide_value,
                "improvement": r.improvement,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "excludes_zero": r.excludes_zero,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_latex(path: Path, results: Sequence[MetricResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired bootstrap 95\\% confidence intervals for the improvements of ArtTIDE over Base on the representative model subset. For CRR, improvement is computed as Base minus ArtTIDE because lower CRR is better.}",
        "\\label{tab:bootstrap_ci_arttide}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Metric & Base & ArtTIDE & Gain (pp) & 95\\% CI (pp) \\\\",
        "\\midrule",
    ]
    current_model = None
    for result in results:
        if current_model is not None and current_model != result.model:
            lines.append("\\midrule")
        current_model = result.model
        lines.append(
            f"{latex_escape(result.model)} & {latex_escape(result.metric)} & "
            f"{result.base_value:.3f} & {result.arttide_value:.3f} & "
            f"{100 * result.improvement:+.2f} & "
            f"[{100 * result.ci_low:+.2f}, {100 * result.ci_high:+.2f}] \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap CI for ArtTIDE vs Base on Table 3 metrics.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "9.2_evaluation_improve" / "bootstrap_outputs"))
    args = parser.parse_args()

    base_module = load_module(BASE_REGISTRY_PATH, "base_eval_registry")
    art_module = load_module(ARTTIDE_REGISTRY_PATH, "arttide_eval_registry")
    base_registry = base_module.MODEL_REGISTRY
    art_registry = art_module.MODEL_REGISTRY

    rng = np.random.default_rng(args.seed)
    all_results: List[MetricResult] = []
    for model_name, base_id, art_id in MODEL_PAIRS:
        base_bundle = load_base_bundle(base_registry[base_id])
        art_bundle = load_arttide_bundle(art_registry[art_id])
        all_results.extend(evaluate_model(model_name, base_bundle, art_bundle, rng, args.n_bootstrap))

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "arttide_vs_base_bootstrap_ci.csv", all_results)
    write_json(output_dir / "arttide_vs_base_bootstrap_ci.json", all_results, args.n_bootstrap, args.seed)
    write_latex(output_dir / "arttide_vs_base_bootstrap_ci_table.tex", all_results)

    print(f"Wrote CSV: {output_dir / 'arttide_vs_base_bootstrap_ci.csv'}")
    print(f"Wrote JSON: {output_dir / 'arttide_vs_base_bootstrap_ci.json'}")
    print(f"Wrote LaTeX: {output_dir / 'arttide_vs_base_bootstrap_ci_table.tex'}")


if __name__ == "__main__":
    main()
