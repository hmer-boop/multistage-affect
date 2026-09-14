from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent

ALLOWED_LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
ALLOWED_LABEL_SET = set(ALLOWED_LABELS)
SPLITTERS = ["/", "、", "|", ",", "，", ";", "；"]


@dataclass
class MetricIO:
    project_root: Path
    gold_base: Path
    gold_round2: Optional[Path]
    pred_stage1: Optional[Path]
    pred_stage2: Optional[Path]
    pred_stage3: Optional[Path]
    pred_stage1_round2: Optional[Path]
    pred_stage2_round2: Optional[Path]
    pred_stage3_round2: Optional[Path]
    output_csv: Optional[Path]
    eval_id_mode: str
    strict: bool


def _resolve_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _stage_match_score(path: Path, stage: str, is_round2: bool) -> int:
    """
    对目录中的候选 JSON/JSONL 文件打分，分数越高越可能是当前 stage 的主结果文件。
    目标是兼容如下多种命名：
    - results_stage1.json
    - results_stage1_round2.json
    - results_stage1_internvl8b.json
    - results_stage1_internvl8b_round2.json
    - 其他包含 stage 关键字的结果文件
    """
    name = path.name.lower()
    stem = path.stem.lower()
    target = f"results_{stage}"
    target_round2 = f"{target}_round2"

    score = 0

    # 精确命名优先
    if stem == target:
        score += 10000
    if stem == target_round2:
        score += 10000

    # 标准前缀优先
    if stem.startswith(target):
        score += 5000

    # 包含 stage 关键字
    if stage.lower() in stem:
        score += 1000

    # round2 条件
    has_round2 = "round2" in stem
    if is_round2 and has_round2:
        score += 800
    if not is_round2 and not has_round2:
        score += 800

    # 常见“结果文件”关键词加分
    if "result" in stem:
        score += 200
    if "results" in stem:
        score += 200

    # 常见干扰文件降分
    bad_tokens = [
        "missing", "report", "metric", "eval", "summary",
        "detail", "manifest", "gold", "illegal", "filter",
        "filtered", "compare", "visual", "visualization",
    ]
    for tok in bad_tokens:
        if tok in stem:
            score -= 1500

    # 非 round2 时，尽量排除 round2 文件；round2 时反之
    if is_round2 and not has_round2:
        score -= 3000
    if not is_round2 and has_round2:
        score -= 3000

    # 文件名越短通常越像主结果文件
    score -= len(name)

    return score


def _resolve_stage_path(raw: Optional[str], project_root: Path, default_dir_name: str, stage: str, is_round2: bool) -> Optional[Path]:
    if raw is None:
        base = project_root / default_dir_name
    else:
        base = Path(raw).expanduser()
        if not base.is_absolute():
            base = (project_root / base).resolve()

    # 用户若直接传入具体文件，则直接用
    if base.suffix.lower() in {".json", ".jsonl"}:
        return base

    # 先尝试历史固定命名
    suffix = f"results_{stage}_round2" if is_round2 else f"results_{stage}"
    exact_candidates = [
        base / f"{suffix}.json",
        base / f"{suffix}.jsonl",
    ]
    for p in exact_candidates:
        if p.exists():
            return p

    # 兼容多种命名格式：从目录里自动挑最像主结果文件的 JSON/JSONL
    if base.exists() and base.is_dir():
        all_jsons = sorted(list(base.glob("*.json")) + list(base.glob("*.jsonl")))
        if all_jsons:
            scored = sorted(
                all_jsons,
                key=lambda p: _stage_match_score(p, stage, is_round2),
                reverse=True,
            )
            best = scored[0]
            if _stage_match_score(best, stage, is_round2) > 0:
                return best

    # 如果都没找到，仍返回默认路径，方便外部报错时看到期望文件名
    return exact_candidates[0]


def _resolve_simple_path(raw: Optional[str], project_root: Path, default_relative: Optional[str]) -> Optional[Path]:
    if raw is None:
        if default_relative is None:
            return None
        return project_root / default_relative
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def build_common_parser(description: str, include_output_csv: bool = False, include_strict: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT), help="项目根目录，默认是当前脚本上一级目录")
    parser.add_argument("--gold-base", default=None, help="gold_label.json 路径")
    parser.add_argument("--gold-round2", default=None, help="gold_label_round2.json 路径，可留空")

    parser.add_argument("--pred-stage1", default=None, help="stage1 预测文件路径，或其所在目录")
    parser.add_argument("--pred-stage2", default=None, help="stage2 预测文件路径，或其所在目录")
    parser.add_argument("--pred-stage3", default=None, help="stage3 预测文件路径，或其所在目录")
    parser.add_argument("--pred-stage1-round2", default=None, help="stage1 round2 预测文件路径，或其所在目录")
    parser.add_argument("--pred-stage2-round2", default=None, help="stage2 round2 预测文件路径，或其所在目录")
    parser.add_argument("--pred-stage3-round2", default=None, help="stage3 round2 预测文件路径，或其所在目录")

    parser.add_argument(
        "--eval-id-mode",
        choices=["gold", "pred"],
        default="gold",
        help="gold=按金标ID全集评估；pred=按预测ID反查金标评估，适合未跑完的模型",
    )

    if include_output_csv:
        parser.add_argument("--output-csv", default=None, help="可选。若提供则导出明细 CSV")

    if include_strict:
        parser.add_argument(
            "--strict",
            choices=["true", "false"],
            default="true",
            help="仅 9.5 使用。true 表示无法比较也记入分母；false 表示跳过",
        )

    return parser


def resolve_common_args(args: argparse.Namespace, include_output_csv: bool = False, include_strict: bool = False) -> MetricIO:
    project_root = Path(args.project_root).expanduser()
    if not project_root.is_absolute():
        project_root = (DEFAULT_PROJECT_ROOT / project_root).resolve()

    gold_base = _resolve_simple_path(args.gold_base, project_root, "results/gold_label.json")
    gold_round2 = _resolve_simple_path(args.gold_round2, project_root, "results/gold_label_round2_eval_legacy.json")

    pred_stage1 = _resolve_stage_path(args.pred_stage1, project_root, "result gpt_stage1", "stage1", False)
    pred_stage2 = _resolve_stage_path(args.pred_stage2, project_root, "result gpt_stage2", "stage2", False)
    pred_stage3 = _resolve_stage_path(args.pred_stage3, project_root, "result gpt_stage3", "stage3", False)
    pred_stage1_round2 = _resolve_stage_path(args.pred_stage1_round2, project_root, "result gpt_stage1", "stage1", True)
    pred_stage2_round2 = _resolve_stage_path(args.pred_stage2_round2, project_root, "result gpt_stage2", "stage2", True)
    pred_stage3_round2 = _resolve_stage_path(args.pred_stage3_round2, project_root, "result gpt_stage3", "stage3", True)

    output_csv = None
    if include_output_csv and getattr(args, "output_csv", None):
        output_csv = Path(args.output_csv).expanduser()
        if not output_csv.is_absolute():
            output_csv = (project_root / output_csv).resolve()

    strict = True
    if include_strict:
        strict = str(getattr(args, "strict", "true")).lower() == "true"

    return MetricIO(
        project_root=project_root,
        gold_base=gold_base,
        gold_round2=gold_round2,
        pred_stage1=pred_stage1,
        pred_stage2=pred_stage2,
        pred_stage3=pred_stage3,
        pred_stage1_round2=pred_stage1_round2,
        pred_stage2_round2=pred_stage2_round2,
        pred_stage3_round2=pred_stage3_round2,
        output_csv=output_csv,
        eval_id_mode=args.eval_id_mode,
        strict=strict,
    )


def check_required_files(paths: list[Optional[Path]], allow_missing: bool = False) -> None:
    missing = []
    for p in paths:
        if p is None:
            if not allow_missing:
                missing.append("<None>")
            continue
        if not p.exists() and not allow_missing:
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("以下文件不存在或未提供:\n" + "\n".join(missing))


def normalize_text(x: Any) -> str:
    return str(x).strip().strip("“”\"' 　\t")


def parse_label_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        if "labels" in x:
            return parse_label_list(x.get("labels"))
        return []
    if isinstance(x, list):
        out: List[str] = []
        for v in x:
            out.extend(parse_label_list(v))
        return out
    s = normalize_text(x)
    if not s:
        return []
    parts = [s]
    for sp in SPLITTERS:
        new_parts: List[str] = []
        for seg in parts:
            new_parts.extend(seg.split(sp))
        parts = new_parts
    return [normalize_text(v) for v in parts if normalize_text(v)]


def is_valid_label_list(labels: List[str]) -> bool:
    return bool(labels) and all(lb in ALLOWED_LABEL_SET for lb in labels)


def parse_valid_label_list(x: Any) -> Optional[List[str]]:
    labels = parse_label_list(x)
    return labels if is_valid_label_list(labels) else None


def extract_stage_value(record: Dict[str, Any], stage: str) -> Any:
    if stage in record:
        return record.get(stage)
    alt = f"{stage}_label"
    if alt in record:
        return record.get(alt)
    for k in record.keys():
        if stage in k:
            return record[k]
    return None


def choose_candidate_ids(gold_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]], mode: str) -> List[str]:
    if mode == "pred":
        return [iid for iid in pred_map.keys() if iid in gold_map]
    return [iid for iid in gold_map.keys() if iid in pred_map]
