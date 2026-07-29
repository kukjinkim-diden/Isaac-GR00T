#!/bin/bash
# GR00T 파인튜닝: HF DidenRobotics/Push_Buttons_Center 의 v2.1_quest_fixed_hand
# 브랜치(오른팔 단일, task_05_press_button_red_right_hand, 56 eps)를 받아 학습한다.
# GR00T 로더는 lerobot v2.1 레이아웃(episodes.jsonl + modality.json)을 읽으므로
# v3.0 브랜치가 아니라 v2.1 브랜치를 쓴다 (v3.0은 lerobot-train 파이프라인용).
#
# eval loss는 훈련 데이터의 10%를 episode 단위로 홀드아웃(EVAL_SPLIT)해 EVAL_STEPS마다
# 계산하고 wandb에 eval_loss로 기록한다. 체크포인트는 SAVE_STEPS_LIST의 각 스텝에 저장되어
# (해당 스텝의 eval_loss와 나란히) eval loss ↔ success rate 상관분석에 쓸 수 있다.
#
# Usage:
#   bash multitask_train.sh                     # 학습 후 HF 업로드
#   MAX_STEPS=50000 bash multitask_train.sh     # 학습 step (기본 100000)
#   EVAL_STEPS=500 bash multitask_train.sh      # eval loss 기록 주기 (기본 1000)
#   EVAL_SPLIT=0.15 bash multitask_train.sh     # eval 홀드아웃 비율 (기본 0.1)
#   SAVE_STEPS_LIST="1000 5000" bash multitask_train.sh  # 체크포인트 스텝 목록
#   PUSH_HF=0 bash multitask_train.sh           # HF 업로드 건너뛰기 (학습만)
#   CKPT=10000 bash multitask_train.sh          # 다른 스텝 체크포인트 업로드 (기본 MAX_STEPS)

set -u

# ── 데이터셋: HF에서 해석 (절대경로 하드코딩 없음) ──────────────────────────
# 부모 리포의 diden_vla/hub_datasets.py 가 HF에서 받아 로컬 경로를 출력한다
# (환경 변수는 diden_vla/lerobot_baselines/_train_common.sh 와 동일한 규약).
#
#   HUB_REPO / HUB_REVISION / HUB_PREFIX / TASKS / DS_BASE
#   DIDEN_DATA_ROOT   hub/ 캐시의 부모 (기본 ~/dataset)
#
# 이 스크립트의 기본값은 Push_Buttons_Center@v2.1_quest_fixed_hand (태스크 폴더가
# 브랜치 루트에 있어 HUB_PREFIX 는 빈 값). 다른 데이터셋은 env 로 겨눈다.
#
# DATASETS 를 직접 넘기면 해석을 건너뛴다 (로컬 실험용).
# 해석기는 GR00T 의 uv 환경으로 실행한다 — 시스템 python3 에는 huggingface_hub 가
# 없을 수 있고, 그 환경은 GR00T 가 베이스 모델을 받을 때 이미 쓰는 것이다.
# RESOLVER_PY 로 교체 가능 (예: RESOLVER_PY="/path/venv/bin/python").
export HUB_REPO="${HUB_REPO:-DidenRobotics/Push_Buttons_Center}"
export HUB_REVISION="${HUB_REVISION:-v2.1_quest_fixed_hand}"
export HUB_PREFIX="${HUB_PREFIX-}"
export TASKS="${TASKS:-task_05_press_button_red_right_hand}"
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_RESOLVER="$(cd "$_HERE/../.." && pwd)/diden_vla/hub_datasets.py"
if [ -z "${DATASETS:-}" ]; then
    if [ ! -f "$_RESOLVER" ]; then
        echo "[FAIL] 해석기 없음: $_RESOLVER"
        echo "       이 스크립트는 DIDEN_Core 서브모듈로 체크아웃된 상태를 전제한다."
        echo "       독립 실행 시에는 DATASETS=\"/path/task_00 /path/task_01 ...\" 로 직접 지정."
        exit 1
    fi
    if [ -n "${RESOLVER_PY:-}" ]; then read -r -a _PY <<< "$RESOLVER_PY"; else _PY=(uv run python); fi
    mapfile -t DATASETS < <("${_PY[@]}" "$_RESOLVER") \
        || { echo "[FAIL] 데이터셋 해석 실패 (HF_TOKEN / HUB_REVISION 확인)"; exit 1; }
    [ ${#DATASETS[@]} -gt 0 ] || { echo "[FAIL] 해석된 데이터셋이 없다"; exit 1; }
else
    read -r -a DATASETS <<< "$DATASETS"
fi

CONFIG=./diden_humanoid_v1_upper_right_arm_hand_config.py  # 오른팔+오른손 config
MAX_STEPS="${MAX_STEPS:-100000}"    # 100k step
# eval loss ↔ success rate 상관분석용 체크포인트 스텝 (비균일)
SAVE_STEPS_LIST="${SAVE_STEPS_LIST:-1000 2000 5000 10000 20000 50000 100000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"    # eval loss 기록 주기 (1000 → 위 체크포인트 스텝 전부 포함)
EVAL_SPLIT="${EVAL_SPLIT:-0.1}"     # 훈련 데이터의 10%를 open-loop eval로 홀드아웃
CKPT="${CKPT:-$MAX_STEPS}"          # 업로드할 체크포인트 스텝
PUSH_HF="${PUSH_HF:-1}"
HF_REPO="${HF_REPO:-DidenRobotics/Humanoid-Upper-Hand-v1}"
OUT_DIR="./checkpoints/diden_humanoid_v1_right_armhand_push_buttons_center"

echo "════════════════════════════════════════════════════════"
echo "  finetune $HUB_REPO@$HUB_REVISION (max_steps=$MAX_STEPS)"
echo "  checkpoints @ steps: $SAVE_STEPS_LIST"
echo "  eval: every $EVAL_STEPS steps, ${EVAL_SPLIT} episode hold-out"
echo "  datasets (${#DATASETS[@]}): "
printf '    %s\n' "${DATASETS[@]}"
echo "  config : $CONFIG"
echo "════════════════════════════════════════════════════════"

# --dataset-path 에 복수 경로를 주면 하나의 mixture 로 합쳐 학습된다
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
    || { echo "[FAIL] train"; exit 1; }

# ── 학습 성공 → HF 업로드 ──────────────────────────────────────────
[ "$PUSH_HF" = "1" ] || { echo "학습 완료 (업로드 생략)"; exit 0; }
ckpt_dir="$OUT_DIR/checkpoint-${CKPT}"
repo_path="humanoid-v1-right_armhand_push_buttons_center"
if [ ! -d "$ckpt_dir" ]; then
    echo "[SKIP] 체크포인트 없음: $ckpt_dir"; exit 1
fi
echo "  upload → $HF_REPO / $repo_path  ($ckpt_dir)"
uv run hf upload "$HF_REPO" "$ckpt_dir" "$repo_path" --repo-type model \
    || { echo "[FAIL] upload"; exit 1; }

echo "════════════════════════════════════════════════════════"
echo "전체 완료 — HF: https://huggingface.co/$HF_REPO/tree/main/$repo_path"
