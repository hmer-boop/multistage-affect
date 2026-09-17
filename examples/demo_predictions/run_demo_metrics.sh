#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO_DIR="$ROOT/examples/demo_predictions"
GOLD="$DEMO_DIR/labels/gold_label.json"
OUT_DIR="$DEMO_DIR/reports/recomputed"

mkdir -p "$OUT_DIR"

run_one_setting() {
  local setting="$1"
  local pred_dir="$DEMO_DIR/predictions/$setting/gpt4o"
  local log="$OUT_DIR/gpt4o_${setting}_metrics.txt"

  {
    echo "===== GPT-4o ${setting} demo metrics ====="
    echo

    python "$ROOT/src/evaluation/base/stage_label_accuracy_tsmr.py" \
      --project-root "$ROOT" \
      --gold-base "$GOLD" \
      --pred-stage1 "$pred_dir/stage1.json" \
      --pred-stage2 "$pred_dir/stage2.json" \
      --pred-stage3 "$pred_dir/stage3.json" \
      --eval-id-mode pred

    echo
    python "$ROOT/src/evaluation/base/three_stage_multi_label_accuracy.py" \
      --project-root "$ROOT" \
      --gold-base "$GOLD" \
      --pred-stage1 "$pred_dir/stage1.json" \
      --pred-stage2 "$pred_dir/stage2.json" \
      --pred-stage3 "$pred_dir/stage3.json" \
      --eval-id-mode pred

    echo
    python "$ROOT/src/evaluation/base/context_supplement_gain_rate.py" \
      --project-root "$ROOT" \
      --gold-base "$GOLD" \
      --pred-stage2 "$pred_dir/stage2.json" \
      --pred-stage3 "$pred_dir/stage3.json" \
      --eval-id-mode pred \
      --output-csv "$OUT_DIR/gpt4o_${setting}_csgr_detail.csv"

    echo
    python "$ROOT/src/evaluation/base/context_regression_rate.py" \
      --project-root "$ROOT" \
      --gold-base "$GOLD" \
      --pred-stage2 "$pred_dir/stage2.json" \
      --pred-stage3 "$pred_dir/stage3.json" \
      --eval-id-mode pred \
      --output-csv "$OUT_DIR/gpt4o_${setting}_crr_detail.csv"
  } | tee "$log"
}

run_one_setting "base"
run_one_setting "cot"
run_one_setting "arttide"

echo
echo "Recomputed metric logs were written to: $OUT_DIR"
