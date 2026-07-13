#!/bin/bash
# GR00T 태스크별 × 정책별(arm/hand) 파인튜닝 일괄 실행 (학습 PC의 gr00t 저장소 루트에서 실행).
# tasks 이름은 dataset 경로의 폴더명(basename)에서 자동 도출된다.
# 팔·손 정책을 분리 학습한다 (공유 flow head 결합 회피). 각 (task × policy) 조합을 순차 실행.
#
# Usage:
#   bash multi_train.sh                 # 전체 태스크 × arm,hand 전부
#   bash multi_train.sh 2 3            # 태스크 인덱스 2,3만 (× arm,hand)
#   POLICIES="arm" bash multi_train.sh # arm 정책만 (task 전부)

set -u

DATASETS=(
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_00_press_button_green
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_01_press_button_red
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_02_press_button_blue
    /workspace/dataset/DidenRobotics/Humanoid_V1_Hand/quest_joystick/task_03_press_button_white
)

# 정책 → modality config 매핑. POLICIES 환경변수로 일부만 선택 가능 (기본 arm hand 둘 다).
declare -A CONFIG=(
    [arm]=./diden_humanoid_v1_upper_left_arm_config.py
    [hand]=./diden_humanoid_v1_upper_left_hand_only_config.py
)
POLICIES="${POLICIES:-arm hand}"

# 인자로 인덱스를 주면 그것만, 없으면 전부
INDICES=("$@")
if [ ${#INDICES[@]} -eq 0 ]; then
    INDICES=($(seq 0 $((${#DATASETS[@]} - 1))))
fi

FAILED=()
for i in "${INDICES[@]}"; do
    ds="${DATASETS[$i]}"
    task="$(basename "$ds")"
    for policy in $POLICIES; do
        cfg="${CONFIG[$policy]}"
        echo "════════════════════════════════════════════════════════"
        echo "  [$i/$policy] finetune: $task"
        echo "  dataset: $ds"
        echo "  config : $cfg"
        echo "════════════════════════════════════════════════════════"
        uv run gr00t/experiment/launch_finetune.py \
            --base-model-path nvidia/GR00T-N1.7-3B \
            --dataset-path "$ds" \
            --embodiment-tag NEW_EMBODIMENT \
            --modality-config-path "$cfg" \
            --num-gpus 1 \
            --output-dir "./checkpoints/diden_humanoid_v1_left_${policy}_${task}" \
            --save-steps 1000 --save-total-limit 5 --max-steps 2000 \
            --global-batch-size 32 --dataloader-num-workers 4 --use-wandb \
            --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
            || { echo "[FAIL] $policy/$task"; FAILED+=("$policy/$task"); }
    done
done

echo "════════════════════════════════════════════════════════"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "전체 완료 (${#INDICES[@]}개 태스크 × [$POLICIES])"
else
    echo "실패: ${FAILED[*]}"
    exit 1
fi
