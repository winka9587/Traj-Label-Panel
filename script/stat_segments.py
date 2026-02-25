import argparse
import json
import logging
import os
from typing import Dict, List, Any, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def load_segments_json(path: str) -> List[Dict[str, Any]]:
    """
    仅加载 segments.json 中的 segments 列表。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError(f"{path} 中的 'segments' 字段格式不正确，应为列表")

    return segments


def analyze_segments(segments: List[Dict[str, Any]], debug: bool = False) -> Tuple[int, int, float, Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    统计：
    - 总 segment 数量（只统计有效的 segment：有有效时间戳且 end_sec > start_sec）
      * 每个 segment 独立统计，segment 之间不会合并或组合
      * 只看 task 级别的 startSec 和 endSec，不依赖 subtask
    - 原始 segment 数量（包括无效的）
    - 所有 segment 的总时长（秒）
    - 每个子任务（按 subtaskId / name 聚合）的总时长和出现次数
    - 无效 segment 列表（用于调试）
    """
    total_segments = 0  # 只统计有效的 segment
    raw_segments_count = len(segments)  # 原始 segment 数量（包括无效的）
    total_duration_sec = 0.0
    invalid_segments = []  # 记录无效的 segment

    # key: subtask_key, value: {"duration_sec": float, "count": int, "desc": str}
    subtask_stats: Dict[str, Dict[str, Any]] = {}

    for idx, seg in enumerate(segments):
        # 统计有效 segment：只看 task 级别的 startSec 和 endSec
        # 重要：每个 segment 独立统计，segment 之间不会合并或组合
        # 即使多个 segment 的时间范围可以合并，也仍然分别计数
        # 一个 task 可能包含多个 subtask，但 segment 计数只基于 task 本身的 startSec 和 endSec
        start_sec = seg.get("startSec")  # task 的开始时间
        end_sec = seg.get("endSec")      # task 的结束时间
        
        if isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)) and end_sec > start_sec:
            # 只统计有效的 segment（基于 task 的 startSec 和 endSec）
            # 注意：
            # 1. 即使 task 包含多个 subtask，也只算作 1 个 segment
            # 2. 每个 segment 独立计数，不会与其他 segment 合并
            total_segments += 1
            total_duration_sec += end_sec - start_sec
        else:
            # 记录无效的 segment
            invalid_segments.append({
                "index": idx,
                "startSec": start_sec,
                "endSec": end_sec,
                "reason": "缺少时间戳" if (start_sec is None or end_sec is None) else "endSec <= startSec",
                "taskId": seg.get("taskId", "unknown"),
            })

        # 统计 subtask 的时长和数量（用于子任务统计，不影响 segment 计数）
        for st in seg.get("subtasks", []) or []:
            st_start = st.get("startSec")
            st_end = st.get("endSec")
            if not (isinstance(st_start, (int, float)) and isinstance(st_end, (int, float)) and st_end > st_start):
                continue

            dur = st_end - st_start
            subtask_id = st.get("subtaskId") or ""
            subtask_name = st.get("name") or ""
            key_parts = [p for p in [subtask_id, subtask_name] if p]
            subtask_key = "/".join(key_parts) if key_parts else "unknown"

            if subtask_key not in subtask_stats:
                subtask_stats[subtask_key] = {
                    "duration_sec": 0.0,
                    "count": 0,
                    "desc": f"id={subtask_id}, name={subtask_name}",
                }

            subtask_stats[subtask_key]["duration_sec"] += dur
            subtask_stats[subtask_key]["count"] += 1

    return total_segments, raw_segments_count, total_duration_sec, subtask_stats, invalid_segments


def pretty_time_from_seconds(seconds: float) -> Tuple[int, int]:
    """
    将秒转为 (小时, 分钟)，向下取整到分钟。
    """
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return hours, minutes


def print_report(
    total_segments: int,
    raw_segments_count: int,
    total_duration_sec: float,
    subtask_stats: Dict[str, Dict[str, Any]],
    invalid_segments: List[Dict[str, Any]] = None,
    debug: bool = False,
) -> None:
    """
    打印统计报告：
    - 总序列数（segment 数）
    - 总时长（小时 + 分钟）
    - 每个子任务的总时长、数量和占比
    """
    total_h, total_m = pretty_time_from_seconds(total_duration_sec)

    logger.info("========== Segment 总体统计 ==========")
    logger.info(f"总序列数（有效 segment 数量）: {total_segments}")
    if raw_segments_count != total_segments:
        invalid_count = raw_segments_count - total_segments
        logger.info(f"原始 segment 数量: {raw_segments_count}（其中 {invalid_count} 个无效，已排除）")
        if debug and invalid_segments:
            logger.info("无效 segment 详情:")
            for inv in invalid_segments[:10]:  # 只显示前10个
                logger.info(f"  - 索引 {inv['index']}: taskId={inv['taskId']}, 原因={inv['reason']}, startSec={inv['startSec']}, endSec={inv['endSec']}")
            if len(invalid_segments) > 10:
                logger.info(f"  ... 还有 {len(invalid_segments) - 10} 个无效 segment")
    logger.info(f"总时长: {total_h} 小时 {total_m} 分钟 (约 {total_duration_sec:.1f} 秒)")

    if not subtask_stats:
        logger.info("未找到任何子任务（subtasks），无法统计子任务时长和占比。")
        return

    total_subtask_duration = sum(v["duration_sec"] for v in subtask_stats.values())
    if total_subtask_duration <= 0:
        logger.info("所有子任务总时长为 0，无法计算占比。")
        return

    logger.info("")
    logger.info("========== 子任务时长与占比（按 subtaskId/name 聚合） ==========")
    logger.info(f"子任务总时长: 约 {total_subtask_duration:.1f} 秒")

    # 按时长从大到小排序
    sorted_items = sorted(
        subtask_stats.items(),
        key=lambda kv: kv[1]["duration_sec"],
        reverse=True,
    )

    for subtask_key, info in sorted_items:
        dur_sec = info["duration_sec"]
        count = info["count"]
        h, m = pretty_time_from_seconds(dur_sec)
        ratio = dur_sec / total_subtask_duration * 100.0
        logger.info(
            f"- 子任务: {subtask_key} | 次数: {count} | 时长: {h} 小时 {m} 分钟 "
            f"(约 {dur_sec:.1f} 秒) | 占比: {ratio:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="仅基于 segments.json 统计 segment / 子任务数量与时长"
    )
    parser.add_argument(
        "--segments",
        type=str,
        default="segments.json",
        help="segments.json 文件路径（默认当前目录下的 segments.json）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出统计结果（用于脚本汇总）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示调试信息（包括无效segment的详细信息）",
    )
    args = parser.parse_args()

    segments = load_segments_json(args.segments)
    total_segments, raw_segments_count, total_duration_sec, subtask_stats, invalid_segments = analyze_segments(segments, debug=args.debug)

    if args.json:
        # JSON 输出模式：输出机器可读的 JSON
        total_subtask_duration = sum(v["duration_sec"] for v in subtask_stats.values())
        output = {
            "total_segments": total_segments,
            "raw_segments_count": raw_segments_count,
            "total_duration_sec": total_duration_sec,
            "subtask_stats": {
                k: {
                    "duration_sec": v["duration_sec"],
                    "count": v["count"],
                }
                for k, v in subtask_stats.items()
            },
            "total_subtask_duration_sec": total_subtask_duration,
        }
        print(json.dumps(output, ensure_ascii=False))
    else:
        # 正常输出模式：打印人类可读的报告
        logger.info(f"读取 segments 文件: {args.segments}")
        print_report(total_segments, raw_segments_count, total_duration_sec, subtask_stats, invalid_segments, debug=args.debug)


if __name__ == "__main__":
    main()


