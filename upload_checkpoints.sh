#!/bin/sh
# 태스크별 GR00T 파인튜닝 체크포인트를 HF 모델 repo에 순차 업로드.
# POSIX sh 호환 (sh/dash/bash 모두 동작). 학습 PC에서 실행.
#
# Usage:
#   sh upload_checkpoints.sh                # 4개 태스크 전부, checkpoint-2000
#   sh upload_checkpoints.sh green white    # 이름 부분일치 태스크만
#   CKPT=1000 sh upload_checkpoints.sh      # 다른 스텝의 체크포인트 업로드

set -u

REPO_ID="DidenRobotics/Humanoid-Upper-Hand-v1"
CKPT_ROOT="/workspace/Isaac-GR00T/checkpoints"
CKPT="${CKPT:-2000}"   # 환경변수로 스텝 오버라이드 가능

TASKS="
task_00_press_button_green
task_01_press_button_red
task_02_press_button_blue
task_03_press_button_white
"

FILTERS="$*"
FAILED=""

for task in $TASKS; do
    if [ -n "$FILTERS" ]; then
        hit=0
        for f in $FILTERS; do
            case "$task" in *"$f"*) hit=1 ;; esac
        done
        [ $hit -eq 1 ] || continue
    fi

    local_path="$CKPT_ROOT/diden_humanoid_v1_left_hand_${task}/checkpoint-${CKPT}"
    repo_path="humanoid-v1-upper-left_hand_${task}"

    if [ ! -d "$local_path" ]; then
        echo "[SKIP] 체크포인트 없음: $local_path"
        FAILED="$FAILED $task(missing)"
        continue
    fi

    echo "════════════════════════════════════════════════════════"
    echo "  upload : $task (checkpoint-${CKPT})"
    echo "  local  : $local_path"
    echo "  in-repo: $repo_path"
    echo "════════════════════════════════════════════════════════"
    hf upload "$REPO_ID" "$local_path" "$repo_path" --repo-type model \
        || { echo "[FAIL] $task"; FAILED="$FAILED $task"; }
done

echo "════════════════════════════════════════════════════════"
if [ -z "$FAILED" ]; then
    echo "전체 업로드 완료 → https://huggingface.co/$REPO_ID"
else
    echo "실패/누락:$FAILED"
    exit 1
fi
