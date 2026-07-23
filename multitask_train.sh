#!/bin/bash
# GR00T 멀티태스크 파인튜닝: 4개 태스크 데이터셋을 하나의 length-weighted mixture로
# 로드해 단일 결합 모델(팔+손, 멀티태스크)을 학습한다.
# 물리적 데이터셋 병합/재업로드 없음 — GR00T가 load 시점에 dataset_paths 리스트를 합쳐준다.
# 각 에피소드의 task annotation이 유지되므로 진짜 멀티태스크 학습.
#
# eval loss는 훈련 데이터의 10%를 episode 단위로 홀드아웃(EVAL_SPLIT)해 EVAL_STEPS마다
# 계산하고 wandb에 eval_loss로 기록한다. 체크포인트는 SAVE_STEPS_LIST의 각 스텝에 저장되어
# (해당 스텝의 eval_loss와 나란히) eval loss ↔ success rate 상관분석에 쓸 수 있다.
#
# Usage:
#   bash multitask_train.sh                     # 4개 태스크 합쳐 학습 후 HF 업로드
#   MAX_STEPS=50000 bash multitask_train.sh     # 학습 step (기본 100000)
#   EVAL_STEPS=500 bash multitask_train.sh      # eval loss 기록 주기 (기본 1000)
#   EVAL_SPLIT=0.15 bash multitask_train.sh     # eval 홀드아웃 비율 (기본 0.1)
#   SAVE_STEPS_LIST="1000 5000" bash multitask_train.sh  # 체크포인트 스텝 목록
#   PUSH_HF=0 bash multitask_train.sh           # HF 업로드 건너뛰기 (학습만)
#   CKPT=10000 bash multitask_train.sh          # 다른 스텝 체크포인트 업로드 (기본 MAX_STEPS)

set -u

DATASETS=(
    /home/diden/dataset/DidenRobotics/Humanoid_V1_Hand/isaac_quest_joystick_fixed_hand/task_00_press_button_red
    /home/diden/dataset/DidenRobotics/Humanoid_V1_Hand/isaac_quest_joystick_fixed_hand/task_01_press_button_blue
    /home/diden/dataset/DidenRobotics/Humanoid_V1_Hand/isaac_quest_joystick_fixed_hand/task_02_press_button_yellow
    /home/diden/dataset/DidenRobotics/Humanoid_V1_Hand/isaac_quest_joystick_fixed_hand/task_03_press_button_white
    /home/diden/dataset/DidenRobotics/Humanoid_V1_Hand/isaac_quest_joystick_fixed_hand/task_04_press_button_green
)

CONFIG=./diden_humanoid_v1_upper_both_arm_hand_config.py  # 팔+손 결합 config
MAX_STEPS="${MAX_STEPS:-100000}"    # 100k step
# eval loss ↔ success rate 상관분석용 체크포인트 스텝 (비균일)
SAVE_STEPS_LIST="${SAVE_STEPS_LIST:-1000 2000 5000 10000 20000 50000 100000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"    # eval loss 기록 주기 (1000 → 위 체크포인트 스텝 전부 포함)
EVAL_SPLIT="${EVAL_SPLIT:-0.1}"     # 훈련 데이터의 10%를 open-loop eval로 홀드아웃
CKPT="${CKPT:-$MAX_STEPS}"          # 업로드할 체크포인트 스텝
PUSH_HF="${PUSH_HF:-1}"
HF_REPO="${HF_REPO:-DidenRobotics/Humanoid-Upper-Hand-v1}"
OUT_DIR="./checkpoints/diden_humanoid_v1_both_armhand_multitask"

echo "════════════════════════════════════════════════════════"
echo "  multitask finetune (max_steps=$MAX_STEPS)"
echo "  checkpoints @ steps: $SAVE_STEPS_LIST"
echo "  eval: every $EVAL_STEPS steps, ${EVAL_SPLIT} episode hold-out"
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
    --save-steps-list $SAVE_STEPS_LIST --max-steps "$MAX_STEPS" \
    --eval-strategy steps --eval-steps "$EVAL_STEPS" \
    --eval-set-split-ratio "$EVAL_SPLIT" --eval-batch-size 2 \
    --global-batch-size 32 --dataloader-num-workers 4 --use-wandb \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    || { echo "[FAIL] multitask train"; exit 1; }

# ── 학습 성공 → HF 업로드 ──────────────────────────────────────────
[ "$PUSH_HF" = "1" ] || { echo "학습 완료 (업로드 생략)"; exit 0; }
ckpt_dir="$OUT_DIR/checkpoint-${CKPT}"
repo_path="humanoid-v1-upper-both_armhand_multitask"
if [ ! -d "$ckpt_dir" ]; then
    echo "[SKIP] 체크포인트 없음: $ckpt_dir"; exit 1
fi
echo "  upload → $HF_REPO / $repo_path  ($ckpt_dir)"
hf upload "$HF_REPO" "$ckpt_dir" "$repo_path" --repo-type model \
    || { echo "[FAIL] upload"; exit 1; }

echo "════════════════════════════════════════════════════════"
echo "전체 완료 — HF: https://huggingface.co/$HF_REPO/tree/main/$repo_path"
