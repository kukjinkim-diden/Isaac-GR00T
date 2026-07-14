#!/bin/bash
# GR00T 멀티태스크 파인튜닝: 4개 태스크 데이터셋을 하나의 length-weighted mixture로
# 로드해 단일 결합 모델(팔+손, 멀티태스크)을 학습한다.
# 물리적 데이터셋 병합/재업로드 없음 — GR00T가 load 시점에 dataset_paths 리스트를 합쳐준다.
# 각 에피소드의 task annotation이 유지되므로 진짜 멀티태스크 학습.
#
# Usage:
#   bash multitask_train.sh                  # 4개 태스크 합쳐 학습 후 HF 업로드
#   MAX_STEPS=30000 bash multitask_train.sh  # 학습 step (기본 20000)
#   PUSH_HF=0 bash multitask_train.sh        # HF 업로드 건너뛰기 (학습만)
#   CKPT=10000 bash multitask_train.sh       # 다른 스텝 체크포인트 업로드 (기본 MAX_STEPS)

set -u

DATASETS=(
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_00_press_button_green
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_01_press_button_red
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_02_press_button_blue
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_03_press_button_white
)

CONFIG=./diden_humanoid_v1_upper_left_arm_hand_config.py  # 팔+손 결합 config
MAX_STEPS="${MAX_STEPS:-40000}"     # 태스크 4배 → 단일 태스크(2000) 대비 넉넉히
SAVE_STEPS="${SAVE_STEPS:-$((MAX_STEPS / 4))}"  # 기본 max_steps/4 (20000 → 5000마다 저장)
CKPT="${CKPT:-$MAX_STEPS}"          # 업로드할 체크포인트 스텝
PUSH_HF="${PUSH_HF:-1}"
HF_REPO="${HF_REPO:-DidenRobotics/Humanoid-Upper-Hand-v1}"
OUT_DIR="./checkpoints/diden_humanoid_v1_left_armhand_multitask"

echo "════════════════════════════════════════════════════════"
echo "  multitask finetune (max_steps=$MAX_STEPS)"
echo "  datasets (${#DATASETS[@]}): "
printf '    %s\n' "${DATASETS[@]}"
echo "  config : $CONFIG"
echo "════════════════════════════════════════════════════════"

# --dataset-path 에 4개 경로를 전달 → 하나의 mixture 로 합쳐 학습
uv run gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "${DATASETS[@]}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$CONFIG" \
    --num-gpus 1 \
    --output-dir "$OUT_DIR" \
    --save-steps "$SAVE_STEPS" --save-total-limit 5 --max-steps "$MAX_STEPS" \
    --global-batch-size 32 --dataloader-num-workers 4 --use-wandb \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    || { echo "[FAIL] multitask train"; exit 1; }

# ── 학습 성공 → HF 업로드 ──────────────────────────────────────────
[ "$PUSH_HF" = "1" ] || { echo "학습 완료 (업로드 생략)"; exit 0; }
ckpt_dir="$OUT_DIR/checkpoint-${CKPT}"
repo_path="humanoid-v1-upper-left_armhand_multitask"
if [ ! -d "$ckpt_dir" ]; then
    echo "[SKIP] 체크포인트 없음: $ckpt_dir"; exit 1
fi
echo "  upload → $HF_REPO / $repo_path  ($ckpt_dir)"
hf upload "$HF_REPO" "$ckpt_dir" "$repo_path" --repo-type model \
    || { echo "[FAIL] upload"; exit 1; }

echo "════════════════════════════════════════════════════════"
echo "전체 완료 — HF: https://huggingface.co/$HF_REPO/tree/main/$repo_path"
