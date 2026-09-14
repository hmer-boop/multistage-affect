from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PYTHON_EXE = sys.executable

# 在这里统一维护模型编号 -> 数据路径
# 以后新增模型，只改这里即可
# 在这里统一维护模型编号 -> 数据路径
# 每个模型、每个阶段都显式指定 base + round2 两份 json
MODEL_REGISTRY: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "GPT-4o",

        "pred_stage1": str(PROJECT_ROOT / "result gpt_stage1" / "results_stage1.json"),
        "pred_stage1_round2": str(PROJECT_ROOT / "result gpt_stage1" / "results_stage1_round2.json"),

        "pred_stage2": str(PROJECT_ROOT / "result gpt_stage2" / "results_stage2.json"),
        "pred_stage2_round2": str(PROJECT_ROOT / "result gpt_stage2" / "results_stage2_round2.json"),

        "pred_stage3": str(PROJECT_ROOT / "result gpt_stage3" / "results_stage3.json"),
        "pred_stage3_round2": str(PROJECT_ROOT / "result gpt_stage3" / "results_stage3_round2.json"),

        "eval_id_mode": "pred",
    },

    2: {
        "name": "Gemini-2.5-Pro",

        "pred_stage1": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro" / "3_batch_rpd_mode"
            / "1_baseline" / "result Gemini_stage1" / "results_stage1.json"
        ),
        "pred_stage1_round2": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro" / "3_batch_rpd_mode"
            / "1_baseline" / "result Gemini_stage1" / "results_stage1_round2.json"
        ),

        "pred_stage2": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro" / "3_batch_rpd_mode"
            / "1_baseline" / "result Gemini_stage2" / "results_stage2.json"
        ),
        "pred_stage2_round2": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro" / "3_batch_rpd_mode"
            / "1_baseline" / "result Gemini_stage2" / "results_stage2_round2.json"
        ),

        "pred_stage3": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro" / "3_batch_rpd_mode"
            / "1_baseline" / "result Gemini_stage3" / "results_stage3.json"
        ),
        "pred_stage3_round2": str(
            PROJECT_ROOT / "000test_gemini" / "gemini 2.5 pro" / "3_batch_rpd_mode"
            / "1_baseline" / "result Gemini_stage3" / "results_stage3_round2.json"
        ),

        "eval_id_mode": "pred",
    },

    3: {
        "name": "InternVL3.5-8B",

        "pred_stage1": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "1_baseline"
            / "result internVL_stage1" / "results_stage1_internvl8b.json"
        ),
        "pred_stage1_round2": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "1_baseline"
            / "result internVL_stage1" / "results_stage1_internvl8b_round2.json"
        ),

        "pred_stage2": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "1_baseline"
            / "result internVL_stage2" / "results_stage2_internvl8b.json"
        ),
        "pred_stage2_round2": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "1_baseline"
            / "result internVL_stage2" / "results_stage2_internvl8b_round2.json"
        ),

        "pred_stage3": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "1_baseline"
            / "result internVL_stage3" / "results_stage3_internvl8b.json"
        ),
        "pred_stage3_round2": str(
            PROJECT_ROOT / "000test_InternVL" / "8b" / "1_baseline"
            / "result internVL_stage3" / "results_stage3_internvl8b_round2.json"
        ),

        "eval_id_mode": "pred",
    },

    4: {
        "name": "InternVL3.5-38B",

        "pred_stage1": str(
            PROJECT_ROOT / "000test_InternVL" / "38b" / "1_baseline"
            / "result internvl_stage1" / "results_stage1_internvl38b.json"
        ),
        "pred_stage1_round2": str(
            PROJECT_ROOT / "000test_InternVL" / "38b" / "1_baseline"
            / "result internvl_stage1" / "results_stage1_internvl38b_round2.json"
        ),

        "pred_stage2": str(
            PROJECT_ROOT / "000test_InternVL" / "38b" / "1_baseline"
            / "result internvl_stage2" / "results_stage2_internvl38b.json"
        ),
        "pred_stage2_round2": str(
            PROJECT_ROOT / "000test_InternVL" / "38b" / "1_baseline"
            / "result internvl_stage2" / "results_stage2_internvl38b_round2.json"
        ),

        "pred_stage3": str(
            PROJECT_ROOT / "000test_InternVL" / "38b" / "1_baseline"
            / "result internvl_stage3" / "results_stage3_internvl38b.json"
        ),
        # 注意：你截图里 38B stage3 的 round2 文件名是 2round，不是 round2
        "pred_stage3_round2": str(
            PROJECT_ROOT / "000test_InternVL" / "38b" / "1_baseline"
            / "result internvl_stage3" / "results_stage3_internvl38b_2round.json"
        ),

        "eval_id_mode": "pred",
    },

    5: {
        "name": "Qwen3-VL-8B",

        "pred_stage1": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "1_baseline"
            / "result qwen_stage1" / "results_stage1_qwen8b.json"
        ),
        "pred_stage1_round2": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "1_baseline"
            / "result qwen_stage1" / "results_stage1_qwen8b_round2.json"
        ),

        "pred_stage2": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "1_baseline"
            / "result qwen_stage2" / "results_stage2_qwen8b.json"
        ),
        "pred_stage2_round2": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "1_baseline"
            / "result qwen_stage2" / "results_stage2_qwen8b_round2.json"
        ),

        # 注意：你截图里 Qwen 8B stage3 文件名没有 8b，是 results_stage3_qwen.json
        "pred_stage3": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "1_baseline"
            / "result qwen_stage3" / "results_stage3_qwen.json"
        ),
        "pred_stage3_round2": str(
            PROJECT_ROOT / "000test_qwen" / "8b-instruct" / "1_baseline"
            / "result qwen_stage3" / "results_stage3_qwen_round2.json"
        ),

        "eval_id_mode": "pred",
    },

    6: {
        "name": "Qwen3-VL-32B",

        "pred_stage1": str(
            PROJECT_ROOT / "000test_qwen" / "32b-instruct" / "1_baseline"
            / "result qwen_stage1" / "results_stage1_qwen32b.json"
        ),
        "pred_stage1_round2": str(
            PROJECT_ROOT / "000test_qwen" / "32b-instruct" / "1_baseline"
            / "result qwen_stage1" / "results_stage1_qwen32b_round2.json"
        ),

        "pred_stage2": str(
            PROJECT_ROOT / "000test_qwen" / "32b-instruct" / "1_baseline"
            / "result qwen_stage2" / "results_stage2_qwen32b.json"
        ),
        "pred_stage2_round2": str(
            PROJECT_ROOT / "000test_qwen" / "32b-instruct" / "1_baseline"
            / "result qwen_stage2" / "results_stage2_qwen32b_round2.json"
        ),

        "pred_stage3": str(
            PROJECT_ROOT / "000test_qwen" / "32b-instruct" / "1_baseline"
            / "result qwen_stage3" / "results_stage3_qwen32b.json"
        ),
        "pred_stage3_round2": str(
            PROJECT_ROOT / "000test_qwen" / "32b-instruct" / "1_baseline"
            / "result qwen_stage3" / "results_stage3_qwen32b_round2.json"
        ),

        "eval_id_mode": "pred",
    },

    7: {
        "name": "LLaVA-OneVision-4B",

        "pred_stage1": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "1_baseline"
            / "result llava_stage1" / "results_stage1_llava4b.json"
        ),
        "pred_stage1_round2": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "1_baseline"
            / "result llava_stage1" / "results_stage1_llava4b_round2.json"
        ),

        "pred_stage2": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "1_baseline"
            / "result llava_stage2" / "results_stage2_llava4b.json"
        ),
        "pred_stage2_round2": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "1_baseline"
            / "result llava_stage2" / "results_stage2_llava4b_round2.json"
        ),

        "pred_stage3": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "1_baseline"
            / "result llava_stage3" / "results_stage3_llava4b.json"
        ),
        # 注意：你截图里 LLaVA 4B stage3 的 round2 文件名是 2round
        "pred_stage3_round2": str(
            PROJECT_ROOT / "000test_llava" / "4b-instruct" / "1_baseline"
            / "result llava_stage3" / "results_stage3_llava4b_2round.json"
        ),

        "eval_id_mode": "pred",
    },

    8: {
        "name": "LLaVA-OneVision-8B",

        "pred_stage1": str(
            PROJECT_ROOT / "000test_llava" / "8b-instruct" / "1_baseline"
            / "result llava_stage1" / "results_stage1_llava8b.json"
        ),
        "pred_stage1_round2": str(
            PROJECT_ROOT / "000test_llava" / "8b-instruct" / "1_baseline"
            / "result llava_stage1" / "results_stage1_llava8b_round2.json"
        ),

        # 注意：你截图里 8B 文件夹下 stage2 文件名仍然写的是 llava4b
        # 如果这确实是 8B 结果，就先按当前真实文件名读取
        "pred_stage2": str(
            PROJECT_ROOT / "000test_llava" / "8b-instruct" / "1_baseline"
            / "result llava_stage2" / "results_stage2_llava8b.json"
        ),
        "pred_stage2_round2": str(
            PROJECT_ROOT / "000test_llava" / "8b-instruct" / "1_baseline"
            / "result llava_stage2" / "results_stage2_llava8b_round2.json"
        ),

        # 注意：你截图里 8B 文件夹下 stage3 文件名也仍然写的是 llava4b
        "pred_stage3": str(
            PROJECT_ROOT / "000test_llava" / "8b-instruct" / "1_baseline"
            / "result llava_stage3" / "results_stage3_llava8b.json"
        ),
        "pred_stage3_round2": str(
            PROJECT_ROOT / "000test_llava" / "8b-instruct" / "1_baseline"
            / "result llava_stage3" / "results_stage3_llava8b_2round.json"
        ),

        "eval_id_mode": "pred",
    },
}

# 在这里统一维护指标编号 -> 子脚本
METRIC_SCRIPTS: Dict[str, Dict[str, Any]] = {
    "9.1": {
        "name": "Stage label accuracy + TSMR",
        "script": "9.1_stage_label_accuracy_tsmr_cli.py",
    },
    "9.2a": {
        "name": "One-stage polarity accuracy",
        "script": "9.2_one_stage_polarity_accuracy_cli.py",
    },
    "9.2b": {
        "name": "Three-stage polarity all consistency",
        "script": "9.2_three_stage_polarity_all_consistency_cli.py",
    },
    "9.3": {
        "name": "3-stage polarity accuracy",
        "script": "9.3_three_stage_polarity_accuracy_cli.py",
    },
    "9.4": {
        "name": "3-stage multi-label accuracy",
        "script": "9.4_three_stage_multi_label_accuracy_cli.py",
    },
    "9.5": {
        "name": "stage2->3 gradient consistency rate",
        "script": "9.5_stage2_to_3_gradient_consistency_rate_cli.py",
    },
    "9.6": {
        "name": "context supplement gain rate",
        "script": "9.6_context_supplement_gain_rate_cli.py",
    },
    "9.7": {
        "name": "Strict Trajectory Accuracy (STA)",
        "script": "9.7_strict_trajectory_accuracy_cli.py",
    },
    "9.8": {
        "name": "Context Regression Rate (CRR)",
        "script": "9.8_context_regression_rate_cli.py",
    },
}

DEFAULT_GOLD_BASE = str(PROJECT_ROOT / "results" / "gold_label.json")
DEFAULT_GOLD_ROUND2 = str(PROJECT_ROOT / "results" / "gold_label_round2_eval_legacy.json")


def list_models() -> None:
    print("===== 可用模型 =====")
    for mid in sorted(MODEL_REGISTRY):
        cfg = MODEL_REGISTRY[mid]
        print(f"{mid}. {cfg['name']} | eval_id_mode={cfg.get('eval_id_mode', 'gold')}")
    print()



def list_metrics() -> None:
    print("===== 可用指标 =====")
    for key, cfg in METRIC_SCRIPTS.items():
        print(f"{key}: {cfg['name']}")
    print("all: 依次运行全部指标")
    print()



def parse_metric_selection(raw: str) -> List[str]:
    raw = raw.strip()
    if raw.lower() == "all":
        return list(METRIC_SCRIPTS.keys())

    parts = [x.strip() for x in raw.replace("，", ",").split(",") if x.strip()]
    selected: List[str] = []
    for p in parts:
        if p not in METRIC_SCRIPTS:
            raise ValueError(f"不支持的指标编号: {p}")
        if p not in selected:
            selected.append(p)
    if not selected:
        raise ValueError("未选择任何有效指标")
    return selected



def build_child_command(model_cfg: Dict[str, Any], metric_key: str) -> List[str]:
    metric_cfg = METRIC_SCRIPTS[metric_key]
    script_path = BASE_DIR / metric_cfg["script"]
    if not script_path.exists():
        raise FileNotFoundError(f"找不到指标脚本: {script_path}")

    cmd = [
        PYTHON_EXE,
        str(script_path),
        "--project-root", str(PROJECT_ROOT),
        "--gold-base", model_cfg.get("gold_base", DEFAULT_GOLD_BASE),
        "--gold-round2", model_cfg.get("gold_round2", DEFAULT_GOLD_ROUND2),
        "--pred-stage1", model_cfg["pred_stage1"],
        "--pred-stage2", model_cfg["pred_stage2"],
        "--pred-stage3", model_cfg["pred_stage3"],
        "--eval-id-mode", model_cfg.get("eval_id_mode", "gold"),
    ]

    if model_cfg.get("pred_stage1_round2"):
        cmd += ["--pred-stage1-round2", model_cfg["pred_stage1_round2"]]
    if model_cfg.get("pred_stage2_round2"):
        cmd += ["--pred-stage2-round2", model_cfg["pred_stage2_round2"]]
    if model_cfg.get("pred_stage3_round2"):
        cmd += ["--pred-stage3-round2", model_cfg["pred_stage3_round2"]]

    # 仅 9.5 需要 strict
    if metric_key == "9.5":
        cmd += ["--strict", str(model_cfg.get("strict", True)).lower()]

    # 某些脚本支持 output-csv，但这里默认不传，保持控制台输出简洁
    return cmd



def run_one_metric(model_id: int, metric_key: str, dry_run: bool = False) -> int:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"模型编号不存在: {model_id}")
    if metric_key not in METRIC_SCRIPTS:
        raise KeyError(f"指标编号不存在: {metric_key}")

    model_cfg = MODEL_REGISTRY[model_id]
    metric_cfg = METRIC_SCRIPTS[metric_key]
    cmd = build_child_command(model_cfg, metric_key)

    print("\n" + "=" * 88)
    print(f"模型: {model_id} - {model_cfg['name']}")
    print(f"指标: {metric_key} - {metric_cfg['name']}")
    print("命令:")
    print(" ".join(f'\"{x}\"' if ' ' in x else x for x in cmd))
    print("=" * 88)

    if dry_run:
        return 0

    sys.stdout.flush()
    completed = subprocess.run(cmd, cwd=str(BASE_DIR))
    return completed.returncode



def interactive_pick_model() -> int:
    list_models()
    raw = input("请输入模型编号: ").strip()
    model_id = int(raw)
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"模型编号不存在: {model_id}")
    return model_id



def interactive_pick_metrics() -> List[str]:
    list_metrics()
    raw = input("请输入指标编号，可用逗号分隔；输入 all 表示全部运行: ").strip()
    return parse_metric_selection(raw)



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="9_evaluation 总控脚本：统一选择模型并调用 9.1~9.8 子指标脚本")
    parser.add_argument("--model", type=int, default=None, help="模型编号，例如 1 / 2 / 3")
    parser.add_argument(
        "--metrics",
        default=None,
        help="指标编号，例如 9.1 或 9.2a,9.4,9.5,9.6,9.7,9.8；输入 all 表示全部运行",
    )
    parser.add_argument("--list-models", action="store_true", help="仅显示模型列表")
    parser.add_argument("--list-metrics", action="store_true", help="仅显示指标列表")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不真正运行")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="若某个指标运行失败，则立即停止，不继续后续指标",
    )
    return parser



def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return
    if args.list_metrics:
        list_metrics()
        return

    model_id = args.model if args.model is not None else interactive_pick_model()
    metrics = parse_metric_selection(args.metrics) if args.metrics is not None else interactive_pick_metrics()

    failed: List[str] = []
    for metric_key in metrics:
        rc = run_one_metric(model_id, metric_key, dry_run=args.dry_run)
        if rc != 0:
            failed.append(metric_key)
            print(f"\n[失败] 指标 {metric_key} 返回码: {rc}")
            if args.stop_on_error:
                break

    print("\n" + "#" * 88)
    print(f"模型 {model_id}: {MODEL_REGISTRY[model_id]['name']}")
    print(f"已请求运行指标: {', '.join(metrics)}")
    if args.dry_run:
        print("本次为 dry-run，仅打印命令，未实际执行。")
    elif failed:
        print(f"失败指标: {', '.join(failed)}")
        success = [m for m in metrics if m not in failed]
        if success:
            print(f"成功指标: {', '.join(success)}")
    else:
        print("全部指标运行完成。")
    print("#" * 88)


if __name__ == "__main__":
    main()
