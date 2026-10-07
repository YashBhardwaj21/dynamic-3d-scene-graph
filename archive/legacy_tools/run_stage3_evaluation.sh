#!/usr/bin/env bash
# Evaluation metrics generator for Stage 3 pose estimation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEQ_NAME="${1:-tum_fr1_desk}"
OUT_DIR="${2:-results/stage3/${SEQ_NAME}}"

if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

cd "$SCRIPT_DIR"

if [ "$SEQ_NAME" = "my_desk_sequence" ]; then
    GT_PATH="data/raw/my_desk_sequence/groundtruth.txt"
else
    GT_PATH="data/raw/rgbd_dataset_freiburg1_desk/groundtruth.txt"
fi

echo "Evaluating Stage 3 pose estimation metrics for: $SEQ_NAME"
echo "Telemetry CSV: ${OUT_DIR}/estimator_state.csv"
echo "Estimated Trajectory: ${OUT_DIR}/estimated.tum"
echo "Ground Truth: $GT_PATH"

python3 tools/evaluate_stage3.py \
    --sequence "$SEQ_NAME" \
    --output_dir "$OUT_DIR" \
    --groundtruth "$GT_PATH" \
    --telemetry_csv "${OUT_DIR}/estimator_state.csv" \
    --estimated_tum "${OUT_DIR}/estimated.tum"

echo "Evaluation complete. Metrics saved in $OUT_DIR."
