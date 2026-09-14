from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class MetricIO:
    project_root: Path
    gold_base: Path
    gold_round2: Path
    pred_stage1: Path
    pred_stage2: Path
    pred_stage3: Path


def _stage_match_score(path: Path, stage: str) -> int:
    name = path.name.lower()
    stem = path.stem.lower()
    target = f"results_{stage}"
    score = 0
    if stem == target:
        score += 10000
    if stem.startswith(target):
        score += 5000
    if stage in stem:
        score += 1000
    if 'result' in stem:
        score += 200
    if 'results' in stem:
        score += 200
    for tok in ['missing', 'report', 'metric', 'eval', 'summary', 'detail', 'manifest', 'gold', 'illegal', 'filter', 'compare', 'visual']:
        if tok in stem:
            score -= 1500
    score -= len(name)
    return score


def _resolve_stage_path(raw: Optional[str], project_root: Path, default_dir_name: str, stage: str) -> Path:
    if raw is None:
        base = project_root / default_dir_name
    else:
        base = Path(raw).expanduser()
        if not base.is_absolute():
            base = (project_root / base).resolve()

    if base.suffix.lower() in {'.json', '.jsonl'}:
        return base

    exact_candidates = [
        base / f'results_{stage}.json',
        base / f'results_{stage}.jsonl',
    ]
    for p in exact_candidates:
        if p.exists():
            return p

    if base.exists() and base.is_dir():
        all_jsons = sorted(list(base.glob('*.json')) + list(base.glob('*.jsonl')))
        if all_jsons:
            scored = sorted(all_jsons, key=lambda p: _stage_match_score(p, stage), reverse=True)
            best = scored[0]
            if _stage_match_score(best, stage) > 0:
                return best

    return exact_candidates[0]


def _resolve_simple_path(raw: Optional[str], project_root: Path, default_relative: str) -> Path:
    if raw is None:
        return project_root / default_relative
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--project-root', default=str(DEFAULT_PROJECT_ROOT), help='项目根目录，默认是当前脚本上一级目录')
    parser.add_argument('--gold-base', default=None, help='gold_label.json 路径')
    parser.add_argument('--gold-round2', default=None, help='gold_label_round2.json 路径')
    parser.add_argument('--pred-stage1', default=None, help='stage1 预测文件路径，或其所在目录')
    parser.add_argument('--pred-stage2', default=None, help='stage2 预测文件路径，或其所在目录')
    parser.add_argument('--pred-stage3', default=None, help='stage3 预测文件路径，或其所在目录')
    return parser


def resolve_common_args(args: argparse.Namespace) -> MetricIO:
    project_root = Path(args.project_root).expanduser()
    if not project_root.is_absolute():
        project_root = (DEFAULT_PROJECT_ROOT / project_root).resolve()

    return MetricIO(
        project_root=project_root,
        gold_base=_resolve_simple_path(args.gold_base, project_root, 'results/gold_label.json'),
        gold_round2=_resolve_simple_path(args.gold_round2, project_root, 'results/gold_label_round2_eval_legacy.json'),
        pred_stage1=_resolve_stage_path(args.pred_stage1, project_root, 'result gpt_stage1', 'stage1'),
        pred_stage2=_resolve_stage_path(args.pred_stage2, project_root, 'result gpt_stage2', 'stage2'),
        pred_stage3=_resolve_stage_path(args.pred_stage3, project_root, 'result gpt_stage3', 'stage3'),
    )


def check_required_files(paths: List[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('以下文件不存在或未提供:\n' + '\n'.join(missing))


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding='utf-8-sig').strip()
    if not text:
        return []
    if '\n' in text and text.lstrip().startswith('{') and text.count('\n{') >= 1:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def rows_from_obj(obj: Any, path: Path) -> List[dict]:
    if isinstance(obj, dict) and isinstance(obj.get('valid'), list):
        rows = obj['valid']
    elif isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = list(obj.values())
    else:
        raise ValueError(f'{path.name} 格式不符合预期：应为列表，或包含 valid 的字典。')
    return [r for r in rows if isinstance(r, dict)]


def normalize_text(x: Any) -> str:
    return str(x).strip().strip('“”\"\' 　\t')


def split_labels(x: Any, splitters: List[str]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        if 'labels' in x:
            return split_labels(x.get('labels'), splitters)
        return []
    if isinstance(x, list):
        out: List[str] = []
        for v in x:
            out.extend(split_labels(v, splitters))
        return [normalize_text(v) for v in out if normalize_text(v)]
    s = normalize_text(x)
    if not s:
        return []
    parts = [s]
    for sp in splitters:
        new_parts: List[str] = []
        for seg in parts:
            new_parts.extend(seg.split(sp))
        parts = new_parts
    return [normalize_text(v) for v in parts if normalize_text(v)]
