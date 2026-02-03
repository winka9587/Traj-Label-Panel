#!/usr/bin/env bash
set -euo pipefail

# ===== Python 解释器 =====
PYTHON_BIN="/home/eiir/miniconda3/envs/pika_convert/bin/python"

# ===== gen 脚本路径 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/gen_lerobot_dataset_full.py"

# ===== 环境变量 =====
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1   # 确保 Python 实时输出
export HF_HOME="${HF_HOME:-/media/eiir/Extreme SSD}"  # HuggingFace 缓存目录，lerobot数据集会保存在此目录下

# ===== 参数 =====
ROOT_DIR="${1:-/media/eiir/Extreme SSD/raw_pika_gbt}"
REPO="${2:-pick_cillion_gbt_v2}"
FPS="${3:-30}"

# CLEAR_MODE:
# 0 = never (default)
# 1 = first time only
# 2 = every time
CLEAR_MODE="${CLEAR_MODE:-0}"

# 标注参数
ANNOTATION_LEVEL="${ANNOTATION_LEVEL:-full_task}"
AVAILABLE_SUBTASK="${AVAILABLE_SUBTASK:-pick place back}"

# ===== 检查 =====
[[ -x "$PYTHON_BIN" ]] || { echo "[ERROR] python not found: $PYTHON_BIN"; exit 1; }
[[ -f "$GEN_SCRIPT" ]] || { echo "[ERROR] gen script not found: $GEN_SCRIPT"; exit 1; }
[[ -d "$ROOT_DIR" ]] || { echo "[ERROR] root dir not found: $ROOT_DIR"; exit 1; }

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] REPO=$REPO FPS=$FPS"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] CLEAR_MODE=$CLEAR_MODE (0=never,1=first,2=always)"
echo "[INFO] ANNOTATION_LEVEL=$ANNOTATION_LEVEL"
echo "[INFO] AVAILABLE_SUBTASK=$AVAILABLE_SUBTASK"
echo

did_clear=0

# ===== 自然数排序（关键改动）=====
mapfile -t DIRS < <(
  find "$ROOT_DIR" -maxdepth 1 -type d -name "train_data_*" \
  | sort -V
)

if [[ ${#DIRS[@]} -eq 0 ]]; then
  echo "[WARN] no train_data_* folders found"
  exit 0
fi

for d in "${DIRS[@]}"; do
  mcap="$(find "$d" -maxdepth 1 -type f -name "*.mcap" | sort -V | head -n 1 || true)"
  [[ -n "$mcap" ]] || { echo "[SKIP] $d : no mcap"; continue; }

  base="${mcap%.mcap}"
  if [[ -f "${base}.json" ]]; then
    json="${base}.json"
  else
    json="$(find "$d" -maxdepth 1 -type f -name "*.json" | sort -V | head -n 1 || true)"
  fi

  [[ -n "${json:-}" ]] || { echo "[SKIP] $d : no json"; continue; }

  echo "================================================="
  echo "[RUN] dir  : $d"
  echo "[RUN] mcap : $mcap"
  echo "[RUN] json : $json"

  clear_arg=()

  case "$CLEAR_MODE" in
    0)
      echo "[RUN] clear: disabled"
      ;;
    1)
      if [[ "$did_clear" -eq 0 ]]; then
        clear_arg=(--clear)
        did_clear=1
        echo "[RUN] clear: first time"
      else
        echo "[RUN] clear: skipped"
      fi
      ;;
    2)
      clear_arg=(--clear)
      echo "[RUN] clear: every time"
      ;;
    *)
      echo "[ERROR] invalid CLEAR_MODE=$CLEAR_MODE (must be 0/1/2)"
      exit 1
      ;;
  esac

  echo "[RUN] command:"
  echo "  $PYTHON_BIN $GEN_SCRIPT --mcap $mcap --segments $json --repo $REPO --fps $FPS --annotation_level $ANNOTATION_LEVEL --available_subtask $AVAILABLE_SUBTASK ${clear_arg[*]}"
  echo

  # ===== 实时显示 Python 输出（stdout + stderr）=====
  echo "[RUN] executing Python script..."
  echo "[DEBUG] Python: $PYTHON_BIN"
  echo "[DEBUG] Script: $GEN_SCRIPT"
  
  # 确保日志文件路径是绝对路径
  LOG_FILE="${SCRIPT_DIR}/gen_dataset.log"
  
  # 临时禁用 pipefail，避免管道命令失败导致脚本退出
  set +o pipefail
  
  # 执行 Python 脚本，输出同时显示和记录
  # 使用 -u 参数强制 Python 无缓冲输出
  if ! "$PYTHON_BIN" -u "$GEN_SCRIPT" \
    --mcap "$mcap" \
    --segments "$json" \
    --repo "$REPO" \
    --fps "$FPS" \
    --annotation_level "$ANNOTATION_LEVEL" \
    --available_subtask $AVAILABLE_SUBTASK \
    "${clear_arg[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    python_exit_code=${PIPESTATUS[0]}
    echo "[ERROR] Python script failed with exit code $python_exit_code" | tee -a "$LOG_FILE"
  else
    python_exit_code=${PIPESTATUS[0]}
    if [[ $python_exit_code -ne 0 ]]; then
      echo "[ERROR] Python script failed with exit code $python_exit_code" | tee -a "$LOG_FILE"
    else
      echo "[RUN] Python script completed successfully." | tee -a "$LOG_FILE"
    fi
  fi
  
  set -o pipefail
  echo ""  # 空行分隔

done

echo
echo "[DONE] all folders processed."
LOG_FILE="${SCRIPT_DIR}/gen_dataset.log"
echo "[LOG] full log saved to: $LOG_FILE"
