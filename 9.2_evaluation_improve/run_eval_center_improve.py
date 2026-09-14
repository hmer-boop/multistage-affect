from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PYTHON_EXE = sys.executable

MODEL_REGISTRY: Dict[int, Dict[str, Any]] = {
    # =========================================================================
    # 1. GPT-4o improve
    # 位置：
    # PythonProject/01improvement method/method data/
    # 特殊说明：
    # GPT-4o improve 不在 result gpt_stage1/2/3 里面，而是在 method data 里
    # =========================================================================
    1: {
        "name": "GPT-4o-improve",

        "gold_base": str(
            PROJECT_ROOT / "results" / "gold_label.json"
        ),
        "gold_round2": str(
            PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json"
        ),

        "pred_stage1": str(
            PROJECT_ROOT / "01improvement method" / "method data"
            / "stage1_gpt_labels4.json"
        ),
        "pred_stage2": str(
            PROJECT_ROOT / "01improvement method" / "method data"
            / "stage2_gpt_labels4.json"
        ),
        "pred_stage3": str(
            PROJECT_ROOT / "01improvement method" / "method data"
            / "stage3_gpt_labels3.json"
        ),
    },

    # =========================================================================
    # 2. Gemini 2.5 Pro improve
    # 位置：
    # PythonProject/000test_gemini/gemini 2.5 pro/3_batch_rpd_mode/2_improve/
    # =========================================================================
    2: {
        "name": "Gemini-2.5-Pro-improve",

        "gold_base": str(
            PROJECT_ROOT / "results" / "gold_label.json"
        ),
        "gold_round2": str(
            PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json"
        ),

        "pred_stage1": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro"
            / "3_batch_rpd_mode" / "2_improve"
            / "result Gemini_stage1" / "stage1_gemini_labels4.json"
        ),
        "pred_stage2": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro"
            / "3_batch_rpd_mode" / "2_improve"
            / "result Gemini_stage2" / "stage2_gemini_labels4.json"
        ),
        "pred_stage3": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro"
            / "3_batch_rpd_mode" / "2_improve"
            / "result Gemini_stage3" / "stage3_gemini_labels3.json"
        ),
    },

    # =========================================================================
    # 3. InternVL3.5-8B improve
    # 位置：
    # PythonProject/000test_InternVL/8b/2_improve/
    # =========================================================================
    3: {
        "name": "InternVL3.5-8B-improve",

        "gold_base": str(
            PROJECT_ROOT / "results" / "gold_label.json"
        ),
        "gold_round2": str(
            PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json"
        ),

        "pred_stage1": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "2_improve"
            / "result internvl_stage1" / "stage1_internvl8b_labels4.json"
        ),
        "pred_stage2": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "2_improve"
            / "result internvl_stage2" / "stage2_internvl8b_labels4.json"
        ),
        "pred_stage3": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "2_improve"
            / "result internvl_stage3" / "stage3_internvl8b_labels3.json"
        ),
    },

    # =========================================================================
    # 4. LLaVA-OneVision-4B improve
    # 位置：
    # PythonProject/000test_llava/4b-instruct/2_improve/
    # =========================================================================
    4: {
        "name": "LLaVA-OneVision-4B-improve",

        "gold_base": str(
            PROJECT_ROOT / "results" / "gold_label.json"
        ),
        "gold_round2": str(
            PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json"
        ),

        "pred_stage1": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "2_improve"
            / "result llava_stage1" / "stage1_llava4b_labels4.json"
        ),
        "pred_stage2": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "2_improve"
            / "result llava_stage2" / "stage2_llava4b_labels4.json"
        ),
        "pred_stage3": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "2_improve"
            / "result llava_stage3" / "stage3_llava4b_labels3.json"
        ),
    },

    # =========================================================================
    # 5. Qwen3-VL-8B improve
    # 位置：
    # PythonProject/000test_qwen/8b-instruct/2_improve/
    # 注意：
    # Qwen improve 文件名不是 stage1_qwen8b_labels4.json，
    # 而是 results_stage1_qwen8b_improve.json 这种命名
    # =========================================================================
    5: {
        "name": "Qwen3-VL-8B-improve",

        "gold_base": str(
            PROJECT_ROOT / "results" / "gold_label.json"
        ),
        "gold_round2": str(
            PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json"
        ),

        "pred_stage1": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "2_improve"
            / "result qwen_stage1" / "results_stage1_qwen8b_improve.json"
        ),
        "pred_stage2": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "2_improve"
            / "result qwen_stage2" / "results_stage2_qwen8b_improve.json"
        ),
        "pred_stage3": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "2_improve"
            / "result qwen_stage3" / "results_stage3_qwen8b_improve.json"
        ),
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


def list_models() -> None:
    print('===== 可用模型 =====')
    for mid in sorted(MODEL_REGISTRY):
        print(f"{mid}. {MODEL_REGISTRY[mid]['name']}")
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


def build_child_command(model_cfg: Dict[str, Any], metric_key: str) -> List[str]:
    metric_cfg = METRIC_SCRIPTS[metric_key]
    script_path = BASE_DIR / metric_cfg['script']
    if not script_path.exists():
        raise FileNotFoundError(f'找不到指标脚本: {script_path}')
    return [
        PYTHON_EXE, str(script_path),
        '--project-root', str(PROJECT_ROOT),
        '--gold-base', model_cfg['gold_base'],
        '--gold-round2', model_cfg['gold_round2'],
        '--pred-stage1', model_cfg['pred_stage1'],
        '--pred-stage2', model_cfg['pred_stage2'],
        '--pred-stage3', model_cfg['pred_stage3'],
    ]


def run_one_metric(model_id: int, metric_key: str, dry_run: bool = False) -> int:
    model_cfg = MODEL_REGISTRY[model_id]
    metric_cfg = METRIC_SCRIPTS[metric_key]
    cmd = build_child_command(model_cfg, metric_key)

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
    parser = argparse.ArgumentParser(description='9.2_evaluation_improve 总控脚本')
    parser.add_argument('--model', type=int, default=None, help='模型编号，例如 1')
    parser.add_argument('--metrics', default=None, help='指标编号，例如 9.1 或 9.2a,9.4,9.5,9.6,9.7,9.8；输入 all 表示全部运行')
    parser.add_argument('--list-models', action='store_true', help='仅显示模型列表')
    parser.add_argument('--list-metrics', action='store_true', help='仅显示指标列表')
    parser.add_argument('--dry-run', action='store_true', help='只打印命令，不实际运行')
    parser.add_argument('--stop-on-error', action='store_true', help='某项失败时立即停止')
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

    metric_keys = parse_metric_selection(args.metrics) if args.metrics else interactive_pick_metrics()

    failed = []
    for key in metric_keys:
        rc = run_one_metric(model_id, key, dry_run=args.dry_run)
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
