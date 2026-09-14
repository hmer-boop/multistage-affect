from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PYTHON_EXE = sys.executable

# COT 结果根目录：
# <PROJECT_ROOT>/000 cot/results
COT_RESULTS_ROOT = PROJECT_ROOT / "000 cot" / "results"
GOLD_BASE = PROJECT_ROOT / "results" / "gold_label.json"
GOLD_ROUND2 = PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json"


def cot_stage_path(model_folder: str, stage: str, round2: bool = False) -> str:
    # 统一读取：000 cot/results/<model_folder>/<stage>/results_<stage>.json
    suffix = f"results_{stage}_round2.json" if round2 else f"results_{stage}.json"
    return str(COT_RESULTS_ROOT / model_folder / stage / suffix)


MODEL_REGISTRY: Dict[int, Dict[str, Any]] = {
    # =========================================================================
    # 1. GPT-4o CoT
    # 截图对应位置：PythonProject/000 cot/results/gpt4o/stage1|stage2|stage3/
    # 读取文件：results_stage1.json / results_stage2.json / results_stage3.json
    # =========================================================================
    1: {
        "name": "GPT-4o-CoT",
        "folder": "gpt4o",
        "gold_base": str(GOLD_BASE),
        "gold_round2": str(GOLD_ROUND2),
        "pred_stage1": cot_stage_path("gpt4o", "stage1"),
        "pred_stage2": cot_stage_path("gpt4o", "stage2"),
        "pred_stage3": cot_stage_path("gpt4o", "stage3"),
    },

    # =========================================================================
    # 2. Gemini 2.5 Pro CoT
    # 截图对应位置：PythonProject/000 cot/results/gemini25pro/stage1|stage2|stage3/
    # 读取文件：results_stage1.json / results_stage2.json / results_stage3.json
    # =========================================================================
    2: {
        "name": "Gemini-2.5-Pro-CoT",
        "folder": "gemini25pro",
        "gold_base": str(GOLD_BASE),
        "gold_round2": str(GOLD_ROUND2),
        "pred_stage1": cot_stage_path("gemini25pro", "stage1"),
        "pred_stage2": cot_stage_path("gemini25pro", "stage2"),
        "pred_stage3": cot_stage_path("gemini25pro", "stage3"),
    },

    # =========================================================================
    # 3. InternVL3.5-8B CoT
    # 截图对应位置：PythonProject/000 cot/results/internvl35_8b/stage1|stage2|stage3/
    # 读取文件：results_stage1.json / results_stage2.json / results_stage3.json
    # =========================================================================
    3: {
        "name": "InternVL3.5-8B-CoT",
        "folder": "internvl35_8b",
        "gold_base": str(GOLD_BASE),
        "gold_round2": str(GOLD_ROUND2),
        "pred_stage1": cot_stage_path("internvl35_8b", "stage1"),
        "pred_stage2": cot_stage_path("internvl35_8b", "stage2"),
        "pred_stage3": cot_stage_path("internvl35_8b", "stage3"),
    },

    # =========================================================================
    # 4. LLaVA-OneVision-1.5-4B CoT
    # 截图对应位置：PythonProject/000 cot/results/llava_onevision_1_5_4b/stage1|stage2|stage3/
    # 读取文件：results_stage1.json / results_stage2.json / results_stage3.json
    # =========================================================================
    4: {
        "name": "LLaVA-OneVision-1.5-4B-CoT",
        "folder": "llava_onevision_1_5_4b",
        "gold_base": str(GOLD_BASE),
        "gold_round2": str(GOLD_ROUND2),
        "pred_stage1": cot_stage_path("llava_onevision_1_5_4b", "stage1"),
        "pred_stage2": cot_stage_path("llava_onevision_1_5_4b", "stage2"),
        "pred_stage3": cot_stage_path("llava_onevision_1_5_4b", "stage3"),
    },

    # =========================================================================
    # 5. Qwen3-VL-8B-Instruct CoT
    # 截图对应位置：PythonProject/000 cot/results/qwen3_vl_8b_instruct/stage1|stage2|stage3/
    # 读取文件：results_stage1.json / results_stage2.json / results_stage3.json
    # =========================================================================
    5: {
        "name": "Qwen3-VL-8B-Instruct-CoT",
        "folder": "qwen3_vl_8b_instruct",
        "gold_base": str(GOLD_BASE),
        "gold_round2": str(GOLD_ROUND2),
        "pred_stage1": cot_stage_path("qwen3_vl_8b_instruct", "stage1"),
        "pred_stage2": cot_stage_path("qwen3_vl_8b_instruct", "stage2"),
        "pred_stage3": cot_stage_path("qwen3_vl_8b_instruct", "stage3"),
    },
}


METRIC_SCRIPTS: Dict[str, Dict[str, str]] = {
    '9.1': {'name': 'Stage label accuracy + TSMR', 'script': '9.1_stage_label_accuracy_tsmr.py'},
    '9.2a': {'name': 'One-stage polarity accuracy', 'script': '9.2a_one_stage_polarity_accuracy.py'},
    '9.2b': {'name': 'Three-stage polarity all consistency', 'script': '9.2b_three_stage_polarity_all_consistency.py'},
    '9.3': {'name': 'Three-stage polarity accuracy', 'script': '9.3_three_stage_polarity_accuracy.py'},
    '9.4': {'name': 'Three-stage multi-label accuracy', 'script': '9.4_three_stage_multi_label_accuracy.py'},
    '9.5': {'name': 'Stage2→3 gradient consistency rate', 'script': '9.5_stage2_to_3_gradient_consistency_rate.py'},
    '9.6': {'name': 'Context supplement gain rate', 'script': '9.6_context_supplement_gain_rate.py'},
    '9.7': {'name': 'Strict Trajectory Accuracy (STA)', 'script': '9.7_strict_trajectory_accuracy.py'},
    '9.8': {'name': 'Context Regression Rate (CRR)', 'script': '9.8_context_regression_rate.py'},
}


def _status(path_str: str) -> str:
    return 'OK' if Path(path_str).exists() else 'MISSING'


def print_model_paths(model_id: int) -> None:
    cfg = MODEL_REGISTRY[model_id]
    print('----- 当前模型明确读取路径 -----')
    print(f"模型编号/名称 : {model_id} - {cfg['name']}")
    print(f"COT模型文件夹 : {COT_RESULTS_ROOT / cfg['folder']}")
    for key in [
        'gold_base', 'gold_round2',
        'pred_stage1', 'pred_stage1_round2',
        'pred_stage2', 'pred_stage2_round2',
        'pred_stage3', 'pred_stage3_round2',
    ]:
        if key not in cfg:
            continue
        path = cfg[key]
        print(f"{key:12s}: [{_status(path)}] {path}")
    print('--------------------------------')


def check_model_paths(model_id: int) -> bool:
    cfg = MODEL_REGISTRY[model_id]
    keys = [
        'gold_base', 'gold_round2',
        'pred_stage1', 'pred_stage1_round2',
        'pred_stage2', 'pred_stage2_round2',
        'pred_stage3', 'pred_stage3_round2',
    ]
    missing = [cfg[k] for k in keys if k in cfg and not Path(cfg[k]).exists()]
    if missing:
        print('路径检查失败，以下文件不存在：')
        for p in missing:
            print(f'- {p}')
        return False
    print('路径检查通过：金标与三阶段预测文件均存在。')
    return True


def list_models() -> None:
    print('===== 可用模型 =====')
    for mid in sorted(MODEL_REGISTRY):
        cfg = MODEL_REGISTRY[mid]
        print(f"{mid}. {cfg['name']}  ->  000 cot/results/{cfg['folder']}/")
    print()


def list_metrics() -> None:
    print('===== 可用指标 =====')
    for key, cfg in METRIC_SCRIPTS.items():
        print(f"{key}: {cfg['name']}")
    print('all: 依次运行全部指标')
    print()


def parse_metric_selection(raw: str) -> List[str]:
    raw = raw.strip()
    if raw.lower() == 'all':
        return list(METRIC_SCRIPTS.keys())
    parts = [x.strip() for x in raw.replace('，', ',').split(',') if x.strip()]
    selected: List[str] = []
    for p in parts:
        if p not in METRIC_SCRIPTS:
            raise ValueError(f'不支持的指标编号: {p}')
        if p not in selected:
            selected.append(p)
    if not selected:
        raise ValueError('未选择任何有效指标')
    return selected


def build_child_command(model_cfg: Dict[str, Any], metric_key: str, quiet_paths: bool = False) -> List[str]:
    metric_cfg = METRIC_SCRIPTS[metric_key]
    script_path = BASE_DIR / metric_cfg['script']
    if not script_path.exists():
        raise FileNotFoundError(f'找不到指标脚本: {script_path}')
    cmd = [
        PYTHON_EXE, str(script_path),
        '--project-root', str(PROJECT_ROOT),
        '--gold-base', model_cfg['gold_base'],
        '--gold-round2', model_cfg['gold_round2'],
        '--pred-stage1', model_cfg['pred_stage1'],
        '--pred-stage2', model_cfg['pred_stage2'],
        '--pred-stage3', model_cfg['pred_stage3'],
    ]
    if model_cfg.get('pred_stage1_round2'):
        cmd += ['--pred-stage1-round2', model_cfg['pred_stage1_round2']]
    if model_cfg.get('pred_stage2_round2'):
        cmd += ['--pred-stage2-round2', model_cfg['pred_stage2_round2']]
    if model_cfg.get('pred_stage3_round2'):
        cmd += ['--pred-stage3-round2', model_cfg['pred_stage3_round2']]
    if quiet_paths:
        cmd.append('--quiet-paths')
    return cmd


def run_one_metric(model_id: int, metric_key: str, dry_run: bool = False, quiet_paths: bool = False) -> int:
    model_cfg = MODEL_REGISTRY[model_id]
    metric_cfg = METRIC_SCRIPTS[metric_key]
    cmd = build_child_command(model_cfg, metric_key, quiet_paths=quiet_paths)

    print('\n' + '=' * 88)
    print(f"模型: {model_id} - {model_cfg['name']}")
    print(f"指标: {metric_key} - {metric_cfg['name']}")
    print('=' * 88)

    if dry_run:
        print(' '.join(f'\"{x}\"' if ' ' in x else x for x in cmd))
        return 0

    sys.stdout.flush()
    completed = subprocess.run(cmd, cwd=str(BASE_DIR))
    return completed.returncode


def interactive_pick_model() -> int:
    list_models()
    raw = input('请输入模型编号: ').strip()
    model_id = int(raw)
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f'模型编号不存在: {model_id}')
    return model_id


def interactive_pick_metrics() -> List[str]:
    list_metrics()
    raw = input('请输入指标编号，可用逗号分隔；输入 all 表示全部运行: ').strip()
    return parse_metric_selection(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='9.4_evaluation_cot 总控脚本')
    parser.add_argument('--model', type=int, default=None, help='模型编号，例如 1')
    parser.add_argument('--metrics', default=None, help='指标编号，例如 9.1 或 9.2a,9.4,9.5,9.6,9.7,9.8；输入 all 表示全部运行')
    parser.add_argument('--list-models', action='store_true', help='仅显示模型列表')
    parser.add_argument('--list-metrics', action='store_true', help='仅显示指标列表')
    parser.add_argument('--dry-run', action='store_true', help='只打印命令，不实际运行')
    parser.add_argument('--check-paths', action='store_true', help='只检查当前模型的金标和三阶段预测文件是否存在')
    parser.add_argument('--stop-on-error', action='store_true', help='某项失败时立即停止')
    parser.add_argument('--quiet-paths', action='store_true', help='子指标脚本不重复打印路径')
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return
    if args.list_metrics:
        list_metrics()
        return

    model_id = args.model if args.model is not None else interactive_pick_model()
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f'模型编号不存在: {model_id}')

    print_model_paths(model_id)

    if args.check_paths:
        ok = check_model_paths(model_id)
        raise SystemExit(0 if ok else 1)

    metric_keys = parse_metric_selection(args.metrics) if args.metrics else interactive_pick_metrics()

    failed = []
    for key in metric_keys:
        rc = run_one_metric(model_id, key, dry_run=args.dry_run, quiet_paths=args.quiet_paths)
        if rc != 0:
            failed.append((key, rc))
            if args.stop_on_error:
                break

    print('\n' + '=' * 88)
    if failed:
        print('运行结束：以下指标失败')
        for key, rc in failed:
            print(f'- {key}: return code = {rc}')
    else:
        print('运行结束：全部指标已完成')
    print('=' * 88)


if __name__ == '__main__':
    main()
