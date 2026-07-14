#!/bin/bash
# GR00T 태스크별 파인튜닝 + HF 업로드 일괄 실행 (gr00t 저장소 루트에서 실행).
# tasks 이름은 dataset 경로의 폴더명(basename)에서 자동 도출된다.
# 단일 결합 모델(팔+손)을 학습한다. 충분한 step으로 학습하면 공유 head로도 팔이 안정적.
#
# Usage:
#   bash multi_train.sh                 # 전체 태스크 학습 후 HF 업로드
#   bash multi_train.sh 2 3            # 태스크 인덱스 2,3만
#   MAX_STEPS=10000 bash multi_train.sh # 학습 step (기본 2000; 결합 모델은 넉넉히)
#   PUSH_HF=0 bash multi_train.sh      # HF 업로드 건너뛰기 (학습만)
#   CKPT=1000 bash multi_train.sh      # 다른 스텝 체크포인트 업로드 (기본 MAX_STEPS)

set -u

DATASETS=(
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_00_press_button_green
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_01_press_button_red
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_02_press_button_blue
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_03_press_button_white
)

CONFIG=./diden_humanoid_v1_upper_left_arm_hand_config.py  # 팔+손 결합 config
MAX_STEPS="${MAX_STEPS:-2000}"      # 학습 step (결합 모델은 넉넉히 — undertraining 시 팔 불안정)
CKPT="${CKPT:-$MAX_STEPS}"          # 업로드할 체크포인트 스텝 (학습 스텝과 기본 동기화)
PUSH_HF="${PUSH_HF:-1}"             # 1이면 학습 성공 후 HF 업로드
HF_REPO="${HF_REPO:-DidenRobotics/Humanoid-Upper-Hand-v1}"

# 인자로 인덱스를 주면 그것만, 없으면 전부
INDICES=("$@")
if [ ${#INDICES[@]} -eq 0 ]; then
    INDICES=($(seq 0 $((${#DATASETS[@]} - 1))))
fi

FAILED=()
for i in "${INDICES[@]}"; do
    ds="${DATASETS[$i]}"
    task="$(basename "$ds")"
    out_dir="./checkpoints/diden_humanoid_v1_left_armhand_${task}"
    echo "════════════════════════════════════════════════════════"
    echo "  [$i] finetune: $task  (max_steps=$MAX_STEPS)"
    echo "  dataset: $ds"
    echo "  config : $CONFIG"
    echo "════════════════════════════════════════════════════════"
    uv run gr00t/experiment/launch_finetune.py \
        --base-model-path nvidia/GR00T-N1.7-3B \
        --dataset-path "$ds" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$CONFIG" \
        --num-gpus 1 \
        --output-dir "$out_dir" \
        --save-steps 1000 --save-total-limit 5 --max-steps "$MAX_STEPS" \
        --global-batch-size 32 --dataloader-num-workers 4 --use-wandb \
        --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
        || { echo "[FAIL] train $task"; FAILED+=("train:$task"); continue; }

    # ── 학습 성공 → HF 업로드 ──────────────────────────────────────────
    [ "$PUSH_HF" = "1" ] || continue
    ckpt_dir="$out_dir/checkpoint-${CKPT}"
    repo_path="humanoid-v1-upper-left_armhand_${task}"
    if [ ! -d "$ckpt_dir" ]; then
        echo "[SKIP] 체크포인트 없음: $ckpt_dir"
        FAILED+=("upload:$task(missing)")
        continue
    fi
    echo "  upload → $HF_REPO / $repo_path  ($ckpt_dir)"
    hf upload "$HF_REPO" "$ckpt_dir" "$repo_path" --repo-type model \
        || { echo "[FAIL] upload $task"; FAILED+=("upload:$task"); }
done

echo "════════════════════════════════════════════════════════"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "전체 완료 (${#INDICES[@]}개 태스크)"
    [ "$PUSH_HF" = "1" ] && echo "HF: https://huggingface.co/$HF_REPO"
else
    echo "실패: ${FAILED[*]}"
    exit 1
fi
