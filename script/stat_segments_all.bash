#!/usr/bin/env bash
set -euo pipefail

# ===== Python 解释器 =====
PYTHON_BIN="/home/eiir/miniconda3/envs/pika_convert/bin/python"

# ===== 脚本路径 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAT_SCRIPT="$SCRIPT_DIR/stat_segments.py"

# ===== 参数 =====
# ROOT_DIR: 包含多个 train_data_* 子目录的根目录
ROOT_DIR="${1:-/media/eiir/Extreme SSD/raw_pika_gbt}"


# ===== 检查 =====
[[ -x "$PYTHON_BIN" ]] || { echo "[ERROR] python not found: $PYTHON_BIN"; exit 1; }
[[ -f "$STAT_SCRIPT" ]] || { echo "[ERROR] stat script not found: $STAT_SCRIPT"; exit 1; }
[[ -d "$ROOT_DIR" ]] || { echo "[ERROR] root dir not found: $ROOT_DIR"; exit 1; }

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] STAT_SCRIPT=$STAT_SCRIPT"
echo

# ===== 自然数排序找到所有 train_data_* 目录 =====
mapfile -t DIRS < <(
  find "$ROOT_DIR" -maxdepth 1 -type d -name "train_data_*" \
  | sort -V
)

if [[ ${#DIRS[@]} -eq 0 ]]; then
  echo "[WARN] no train_data_* folders found under $ROOT_DIR"
  exit 0
fi

# 临时文件用于收集 JSON 输出
TMP_JSON_FILE=$(mktemp)
trap "rm -f '$TMP_JSON_FILE'" EXIT

for d in "${DIRS[@]}"; do
  # 在每个子目录中寻找 json（优先与 mcap 同名的 json，其次任意 json）
  mcap="$(find "$d" -maxdepth 1 -type f -name "*.mcap" | sort -V | head -n 1 || true)"

  json=""
  if [[ -n "$mcap" ]]; then
    base="${mcap%.mcap}"
    if [[ -f "${base}.json" ]]; then
      json="${base}.json"
    fi
  fi

  if [[ -z "$json" ]]; then
    json="$(find "$d" -maxdepth 1 -type f -name "*.json" | sort -V | head -n 1 || true)"
  fi

  [[ -n "${json:-}" ]] || { echo "[SKIP] $d : no json"; continue; }

  echo "================================================="
  echo "[RUN] dir   : $d"
  echo "[RUN] json  : $json"
  echo

  # 调用 Python 统计脚本，同时显示输出和收集 JSON
  # 先获取 JSON 输出并保存
  json_output="$("$PYTHON_BIN" -u "$STAT_SCRIPT" --segments "$json" --json 2>/dev/null)"
  
  if [[ -n "$json_output" ]]; then
    echo "$json_output" >> "$TMP_JSON_FILE"
  fi
  
  # 也显示人类可读的输出
  "$PYTHON_BIN" -u "$STAT_SCRIPT" --segments "$json"

  echo ""  # 空行分隔
done

echo
echo "================================================="
echo "========== 所有文件夹汇总统计 =========="
echo "================================================="

# 使用 Python 汇总所有 JSON 数据
"$PYTHON_BIN" -c "
import json
import sys
from collections import defaultdict

# 读取所有 JSON 行
all_stats = []
try:
    with open('$TMP_JSON_FILE', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_stats.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
except FileNotFoundError:
    pass

if not all_stats:
    print('没有找到任何统计数据')
    sys.exit(0)

# 汇总统计
total_segments = sum(s.get('total_segments', 0) for s in all_stats)
total_duration_sec = sum(s.get('total_duration_sec', 0.0) for s in all_stats)
total_subtask_duration_sec = sum(s.get('total_subtask_duration_sec', 0.0) for s in all_stats)

# 汇总子任务统计
subtask_duration = defaultdict(float)
subtask_count = defaultdict(int)

for stats in all_stats:
    for key, info in stats.get('subtask_stats', {}).items():
        subtask_duration[key] += info.get('duration_sec', 0.0)
        subtask_count[key] += info.get('count', 0)

# 格式化输出
def pretty_time(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return hours, minutes

total_h, total_m = pretty_time(total_duration_sec)
print(f'总序列数（所有 segment 数量之和）: {total_segments}')
print(f'总时长（所有 segment 时长之和）: {total_h} 小时 {total_m} 分钟 (约 {total_duration_sec:.1f} 秒)')

if total_subtask_duration_sec > 0:
    print('')
    print('========== 所有子任务汇总统计（按 subtaskId/name 聚合） ==========')
    print(f'子任务总时长: 约 {total_subtask_duration_sec:.1f} 秒')
    print('')
    
    # 按时长排序
    sorted_subtasks = sorted(subtask_duration.items(), key=lambda x: x[1], reverse=True)
    
    for key, dur in sorted_subtasks:
        count = subtask_count[key]
        h, m = pretty_time(dur)
        ratio = (dur / total_subtask_duration_sec * 100.0) if total_subtask_duration_sec > 0 else 0.0
        print(f'- 子任务: {key} | 次数: {count} | 时长: {h} 小时 {m} 分钟 (约 {dur:.1f} 秒) | 占比: {ratio:.2f}%')
"

echo
echo "[DONE] all folders processed."


