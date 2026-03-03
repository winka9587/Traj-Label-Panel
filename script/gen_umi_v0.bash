#!/usr/bin/env bash
set -euo pipefail

# 生成umi数据集版本umi_v0
# 整个序列使用相同的prompt，不拆分子任务
# 目标仅使用右手画面（数据集中左右手均包含）
# 支持多个源数据目录，默认包含：
#   /media/eiir/Extreme SSD/umi_raw_data_part2/pika_data
#   /media/eiir/Extreme SSD/umi_raw_data_part2/pika_data_2/dataset_1
#   /media/eiir/Extreme SSD/umi_raw_data_part2/pika_data_2/dataset_2

# ===== Python 解释器 =====
PYTHON_BIN="/home/eiir/miniconda3/envs/pika_convert/bin/python"

# ===== gen 脚本路径 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/gen_lerobot_dataset_full.py"
# ===== topic 配置文件（与 script 同级的 config 目录）=====
TOPIC_CONFIG="$(cd "$SCRIPT_DIR/.." && pwd)/config/pika_umi_config.json"

# ===== 环境变量 =====
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1   # 确保 Python 实时输出
export HF_HOME="${HF_HOME:-/media/eiir/Extreme SSD}"  # HuggingFace 缓存目录，生成的lerobot数据集会保存在此目录下

# ===== 参数：支持多个源数据目录 =====
# 用法：无参则使用下方默认的 ROOT_DIRS；有参则前 N 个为目录、其后为 REPO 和 FPS
# 例：gen_umi_v0.bash
# 例：gen_umi_v0.bash /path/to/root1
# 例：gen_umi_v0.bash /path/to/root1 /path/to/root2
# 例：gen_umi_v0.bash /path/to/root1 /path/to/root2 pick_cillion_umi_v0 30

# raw数据目录
BASE="/media/eiir/Extreme SSD/umi_raw_data_part2"
# 可通过注释快速启用/关闭
DEFAULT_ROOT_DIRS=(
  "$BASE/pika_data"  
  "$BASE/pika_data_2/dataset_1"
  "$BASE/pika_data_2/dataset_2"
)

# 可修改：episode 子文件夹名称前缀（find -name 的 glob）
EPISODE_DIR_PREFIXES=(
  "train_data_*"
  "ep_*"
  "fp_*"
)

ROOT_DIRS=()
if [[ $# -eq 0 ]]; then
  ROOT_DIRS=("${DEFAULT_ROOT_DIRS[@]}")
  REPO="${REPO:-pick_cillion_umi_v0}"
  FPS="${FPS:-30}"
else
  # 收集所有前导的“目录”参数
  idx=1
  while [[ idx -le $# ]] && [[ -d "${!idx}" ]]; do
    ROOT_DIRS+=("${!idx}")
    ((idx++)) || true
  done
  # 若一个目录都没传，则用默认目录列表
  if [[ ${#ROOT_DIRS[@]} -eq 0 ]]; then
    ROOT_DIRS=("${DEFAULT_ROOT_DIRS[@]}")
  fi
  REPO="${!idx:-pick_cillion_umi_v0}"
  ((idx++)) || true
  FPS="${!idx:-30}"
fi

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
[[ -f "$TOPIC_CONFIG" ]] || { echo "[ERROR] topic config not found: $TOPIC_CONFIG"; exit 1; }
for _r in "${ROOT_DIRS[@]}"; do
  [[ -d "$_r" ]] || { echo "[ERROR] root dir not found: $_r"; exit 1; }
done

echo "[INFO] ROOT_DIRS=(${ROOT_DIRS[*]})"
echo "[INFO] TOPIC_CONFIG=$TOPIC_CONFIG"
echo "[INFO] EPISODE_DIR_PREFIXES=(${EPISODE_DIR_PREFIXES[*]})"
echo "[INFO] REPO=$REPO FPS=$FPS"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] CLEAR_MODE=$CLEAR_MODE (0=never,1=first,2=always)"
echo "[INFO] ANNOTATION_LEVEL=$ANNOTATION_LEVEL"
echo "[INFO] AVAILABLE_SUBTASK=$AVAILABLE_SUBTASK"
echo

did_clear=0

# ===== 从多个源目录收集 episode 目录（含子目录），自然数排序 =====
if [[ ${#EPISODE_DIR_PREFIXES[@]} -eq 0 ]]; then
  echo "[WARN] EPISODE_DIR_PREFIXES is empty"
  exit 0
fi
mapfile -t DIRS < <(
  for _root in "${ROOT_DIRS[@]}"; do
    _find_names=()
    for _p in "${EPISODE_DIR_PREFIXES[@]}"; do
      [[ ${#_find_names[@]} -gt 0 ]] && _find_names+=(-o)
      _find_names+=(-name "$_p")
    done
    find "$_root" -type d \( "${_find_names[@]}" \)
  done | sort -V
)

if [[ ${#DIRS[@]} -eq 0 ]]; then
  echo "[WARN] no episode folders (${EPISODE_DIR_PREFIXES[*]}) found under: ${ROOT_DIRS[*]}"
  echo "[DEBUG] list one level under first root:"
  _first="${ROOT_DIRS[0]}"
  ls -la "$_first" 2>/dev/null || true
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
  echo "  $PYTHON_BIN $GEN_SCRIPT --mcap $mcap --segments $json --repo $REPO --fps $FPS --annotation_level $ANNOTATION_LEVEL --available_subtask $AVAILABLE_SUBTASK --config $TOPIC_CONFIG ${clear_arg[*]}"
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
    --config "$TOPIC_CONFIG" \
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
  
  # 检查是否出现磁盘空间不足错误
  # 检查最近执行的输出中是否包含磁盘空间不足错误
  if tail -n 100 "$LOG_FILE" 2>/dev/null | grep -qiE "(No space left on device|OSError.*\[Errno 28\])"; then
    echo "[FATAL] 检测到磁盘空间不足错误 (OSError: [Errno 28] No space left on device)" | tee -a "$LOG_FILE"
    echo "[FATAL] 中断后续执行" | tee -a "$LOG_FILE"
    set -o pipefail
    exit 1
  fi
  
  set -o pipefail
  echo ""  # 空行分隔

done

echo
echo "[DONE] all folders processed."
LOG_FILE="${SCRIPT_DIR}/gen_dataset.log"
echo "[LOG] full log saved to: $LOG_FILE"
