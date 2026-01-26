# 使用mcap文件和标签文件json（segments.json），生成lerobot数据集
# 1. 读取mcap文件，提取画面和topic数据
# 2. 读取每个mcap文件对应的标签文件json，提取整个mcap轨迹对应的多个segment的start和end时间戳
# 3. 对每个segment，截取对应的画面和topic数据，保存为lerobot数据集格式

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import cv2
import hashlib
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 如果没有tqdm，创建一个简单的占位符
    def tqdm(iterable, *args, **kwargs):
        return iterable

from utils.load_mcap import read_mcap_file, read_pose_from_mcap, load_mcap_data
from utils.load_json import load_segments_json, extract_segment_data
from utils.data_sync import sync_topic_data


def calculate_file_hash(file_path: str) -> str:
    """
    计算文件的SHA256哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件的十六进制哈希值
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 分块读取，避免大文件占用过多内存
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_processed_files_path(output_path: Optional[str], repo_name: str) -> Path:
    """
    获取处理记录JSON文件的路径
    
    Args:
        output_path: 输出路径（可选）
        repo_name: lerobot数据集repo名称
        
    Returns:
        处理记录JSON文件的路径
    """
    if output_path is None:
        base_path = HF_LEROBOT_HOME
    else:
        base_path = Path(output_path)
    
    return base_path / "processed_files.json"


def load_processed_files(output_path: Optional[str], repo_name: str) -> Dict[str, Any]:
    """
    加载已处理文件记录
    
    Args:
        output_path: 输出路径（可选）
        repo_name: lerobot数据集repo名称
        
    Returns:
        已处理文件记录的字典
    """
    processed_files_path = get_processed_files_path(output_path, repo_name)
    
    if not processed_files_path.exists():
        return {}
    
    try:
        with open(processed_files_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法读取处理记录文件 {processed_files_path}: {e}")
        return {}


def save_processed_files(
    output_path: Optional[str],
    repo_name: str,
    processed_files: Dict[str, Any]
) -> None:
    """
    保存已处理文件记录
    
    Args:
        output_path: 输出路径（可选）
        repo_name: lerobot数据集repo名称
        processed_files: 已处理文件记录的字典
    """
    processed_files_path = get_processed_files_path(output_path, repo_name)
    
    # 确保目录存在
    processed_files_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(processed_files_path, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"警告: 无法保存处理记录文件 {processed_files_path}: {e}")


def check_already_processed(
    mcap_path: str,
    segments_json_path: str,
    output_path: Optional[str],
    repo_name: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    检查文件组合是否已经处理过
    
    Args:
        mcap_path: mcap文件路径
        segments_json_path: segments.json文件路径
        output_path: 输出路径（可选）
        repo_name: lerobot数据集repo名称
        
    Returns:
        (是否已处理, 处理记录信息)
    """
    # 计算文件哈希
    try:
        mcap_hash = calculate_file_hash(mcap_path)
        segments_hash = calculate_file_hash(segments_json_path)
    except Exception as e:
        print(f"警告: 无法计算文件哈希: {e}")
        return False, None
    
    # 组合哈希作为唯一标识
    combined_hash = f"{mcap_hash}_{segments_hash}"
    
    # 加载处理记录
    processed_files = load_processed_files(output_path, repo_name)
    
    # 检查是否已处理
    if combined_hash in processed_files:
        record = processed_files[combined_hash]
        return True, record
    
    return False, None


def update_processed_files(
    mcap_path: str,
    segments_json_path: str,
    output_path: Optional[str],
    repo_name: str,
    in_lerobot: bool = True
) -> None:
    """
    更新已处理文件记录
    
    Args:
        mcap_path: mcap文件路径
        segments_json_path: segments.json文件路径
        output_path: 输出路径（可选）
        repo_name: lerobot数据集repo名称
        in_lerobot: 是否已经在lerobot数据集中
    """
    # 计算文件哈希
    try:
        mcap_hash = calculate_file_hash(mcap_path)
        segments_hash = calculate_file_hash(segments_json_path)
    except Exception as e:
        print(f"警告: 无法计算文件哈希: {e}")
        return
    
    # 组合哈希作为唯一标识
    combined_hash = f"{mcap_hash}_{segments_hash}"
    
    # 加载现有记录
    processed_files = load_processed_files(output_path, repo_name)
    
    # 更新记录
    processed_files[combined_hash] = {
        "mcap_path": mcap_path,
        "segments_json_path": segments_json_path,
        "mcap_hash": mcap_hash,
        "segments_hash": segments_hash,
        "repo_name": repo_name,
        "in_lerobot": in_lerobot,
        "processed_time": datetime.now().isoformat()  # 使用当前时间作为处理时间戳
    }
    
    # 保存记录
    save_processed_files(output_path, repo_name, processed_files)


def print_topic_data_summary(topic_data: Dict[str, Dict[str, List[Tuple[float, Any]]]]):
    for data_type, topics_data in topic_data.items():
        for topic_name, data_list in topics_data.items():
            if len(data_list) > 1:
                start_time = data_list[0][0]
                end_time = data_list[-1][0]
                time_span = end_time - start_time
                fps = len(data_list) / time_span if time_span > 0 else 0.0
                print(f"已加载 {data_type} topic '{topic_name}'，数据量: {len(data_list)}，时间范围: {time_span:.2f}s，帧率: {fps:.2f} Hz")
            else:
                print(f"已加载 {data_type} topic '{topic_name}'，数据量: {len(data_list)}")


def create_lerobot_dataset(
    mcap_path: str,
    segments_json_path: str,
    repo_name: str,
    output_path: Optional[str] = None,
    fps: int = 50,
    clear_dataset: bool = False,
    action_type: str = 'current_state'
) -> str:
    """
    从mcap文件和segments.json创建lerobot数据集
    
    Args:
        mcap_path: mcap文件路径
        segments_json_path: segments.json文件路径
        repo_name: lerobot数据集repo名称
        output_path: 输出路径（可选）
        fps: 目标帧率
        clear_dataset: 是否清空已存在的数据集
        
    Returns:
        处理日志
    """
    logs = []
    
    # 检查文件是否存在
    if not os.path.isfile(mcap_path):
        return f"错误: mcap文件不存在: {mcap_path}"
    
    if not os.path.isfile(segments_json_path):
        return f"错误: segments.json文件不存在: {segments_json_path}"
    
    # 检查是否已经处理过
    already_processed, record = check_already_processed(mcap_path, segments_json_path, output_path, repo_name)
    if already_processed and record:
        if record.get('in_lerobot', False):
            logs.append(f"文件组合已处理过，且已在lerobot数据集中，跳过处理")
            logs.append(f"  mcap文件: {record.get('mcap_path', mcap_path)}")
            logs.append(f"  segments文件: {record.get('segments_json_path', segments_json_path)}")
            logs.append(f"  repo名称: {record.get('repo_name', repo_name)}")
            return "\n".join(logs)
        else:
            logs.append(f"文件组合已处理过，但未在lerobot中，继续处理...")
    
    logs.append(f"读取mcap文件: {mcap_path}")
    logs.append(f"读取segments文件: {segments_json_path}")
    
    # key是type, value是topic列表
    topic_list = {
        "pose7d": ["/jbt_arm_R/current_arm_end_pose", "/jbt_arm_L/current_arm_end_pose"],
        "joint_states": ["/jbt_arm_R/current_arm_joint_state", "/jbt_arm_L/current_arm_joint_state"],
        "images": ["/gripper/camera_fisheye_l/color/image_raw", "/gripper/camera_fisheye_r/color/image_raw"],
        "gripper": ["/gripper/gripper_l/data", "/gripper/gripper_r/data"],
    }
    
    # 先读取segments.json，收集所有需要的时间范围
    try:
        segments = load_segments_json(segments_json_path)
        logs.append(f"成功读取segments.json，找到 {len(segments)} 个segment")
    except Exception as e:
        return f"读取segments.json失败: {e}"
    
    # 收集所有需要的时间范围（包括segment和subtask）
    time_ranges = []
    segment_info_list = []  # 存储每个segment/subtask的信息
    
    for idx, segment in enumerate(segments):
        start_sec = segment.get('startSec')
        end_sec = segment.get('endSec')
        prompt = segment.get('prompt', '')
        taskId = segment.get('taskId', 'unknown')
        
        if start_sec is None or end_sec is None:
            continue
        if end_sec <= start_sec:
            continue
        
        subtasks = segment.get('subtasks', [])
        if len(subtasks) == 0:
            # 没有subtasks，使用segment本身
            time_ranges.append((start_sec, end_sec))
            segment_info_list.append({
                'type': 'segment',
                'segment_idx': idx,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'prompt': prompt,
                'taskId': taskId,
            })
        else:
            # 有subtasks，收集每个subtask的时间范围
            for subtask_meta in subtasks:
                subtask_start_sec = subtask_meta.get('startSec', None)
                subtask_end_sec = subtask_meta.get('endSec', None)
                if subtask_start_sec is None or subtask_end_sec is None:
                    continue
                if subtask_end_sec <= subtask_start_sec:
                    continue
                
                time_ranges.append((subtask_start_sec, subtask_end_sec))
                segment_info_list.append({
                    'type': 'subtask',
                    'segment_idx': idx,
                    'subtaskId': subtask_meta.get('subtaskId', 'unknown'),
                    'subtaskName': subtask_meta.get('subtaskName', 'unknown'),
                    'subtask_prompt': subtask_meta.get('prompt', ''),
                    'start_sec': subtask_start_sec,
                    'end_sec': subtask_end_sec,
                    'prompt': prompt,
                    'taskId': taskId,
                })
    
    if not time_ranges:
        return "错误: 没有有效的时间范围"
    
    # 合并重叠的时间范围，计算需要加载的总时间范围
    time_ranges.sort(key=lambda x: x[0])
    merged_ranges = []
    current_start, current_end = time_ranges[0]
    
    for start, end in time_ranges[1:]:
        if start <= current_end:
            # 重叠，合并
            current_end = max(current_end, end)
        else:
            # 不重叠，保存当前范围，开始新的范围
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    merged_ranges.append((current_start, current_end))
    
    # 计算总时间范围
    overall_start = min(r[0] for r in merged_ranges)
    overall_end = max(r[1] for r in merged_ranges)
    
    logs.append(f"收集到 {len(segment_info_list)} 个有效的时间段")
    logs.append(f"合并后需要加载的时间范围: {overall_start:.2f}s - {overall_end:.2f}s (共 {overall_end - overall_start:.2f}s)")
    
    # 一次性加载所有需要的数据
    try:
        logs.append("从mcap文件一次性加载数据...")
        all_topic_data = load_mcap_data(mcap_path, topic_list, 
                                       start_time=overall_start, 
                                       end_time=overall_end)
        logs.append("数据加载完成")
    except Exception as e:
        return f"读取mcap文件失败: {e}"
    
    # 确定输出路径
    if output_path is None:
        lerobot_output_path = HF_LEROBOT_HOME / repo_name
    else:
        lerobot_output_path = Path(output_path) / repo_name
    
    # 清空已存在的数据集
    if clear_dataset and lerobot_output_path.exists():
        import shutil
        shutil.rmtree(lerobot_output_path)
        logs.append(f"已清空已存在的数据集: {lerobot_output_path}")
        
        # 如果清空数据集，也清除相关的处理记录
        try:
            already_processed, record = check_already_processed(mcap_path, segments_json_path, output_path, repo_name)
            if already_processed:
                # 计算文件哈希
                mcap_hash = calculate_file_hash(mcap_path)
                segments_hash = calculate_file_hash(segments_json_path)
                combined_hash = f"{mcap_hash}_{segments_hash}"
                
                # 加载并删除记录
                processed_files = load_processed_files(output_path, repo_name)
                if combined_hash in processed_files:
                    del processed_files[combined_hash]
                    save_processed_files(output_path, repo_name, processed_files)
                    logs.append(f"已清除相关的处理记录")
        except Exception as e:
            logs.append(f"警告: 清除处理记录失败: {e}")
    
    # 创建或加载lerobot数据集
    lerobot_exists = lerobot_output_path.exists()
    if lerobot_exists:
        logs.append(f"数据集已存在，追加新episode: {lerobot_output_path}")
        dataset = LeRobotDataset(repo_id=repo_name)
        dataset.start_image_writer(
            num_processes=5,
            num_threads=10,
        )
    else:
        logs.append(f"创建新数据集: {lerobot_output_path}")
        # 图像尺寸：参考 refer_create_lerobot.py，使用 (640, 480, 3) 即 (width, height, channel)
        image_shape = (640, 480, 3)  # 默认尺寸，参考 refer_create_lerobot.py
        
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type="gbt",
            fps=fps,
            features={
                "wrist_image_left": {
                    "dtype": "image",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                },
                "wrist_image_right": {
                    "dtype": "image",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    
    # 处理每个segment/subtask
    processed_count = 0
    skipped_count = 0
    
    for seg_info in segment_info_list:
        start_sec = seg_info['start_sec']
        end_sec = seg_info['end_sec']
        
        if seg_info['type'] == 'subtask':
            log_prefix = f"Segment {seg_info['segment_idx']} Subtask {seg_info['subtaskId']}"
        else:
            log_prefix = f"Segment {seg_info['segment_idx']}"
        
        logs.append(f"处理{log_prefix}: {start_sec:.2f}s - {end_sec:.2f}s")
        
        try:
            # 从已加载的数据中裁剪当前segment/subtask的时间范围
            segment_topic_data = {}
            for data_type, topics_data in all_topic_data.items():
                segment_topic_data[data_type] = {}
                for topic_name, data_list in topics_data.items():
                    # 过滤出当前时间范围内的数据
                    filtered_data = [(ts, data) for ts, data in data_list 
                                    if start_sec <= ts <= end_sec]
                    if filtered_data:
                        segment_topic_data[data_type][topic_name] = filtered_data
            
            # 对裁剪后的数据进行同步和插值
            logs.append(f"{log_prefix}: 开始数据同步&插值...")
            segment_topic_data = sync_topic_data(segment_topic_data, time_diff_limit=0.03)
            print_topic_data_summary(segment_topic_data)
            logs.append(f"{log_prefix}: 数据同步&插值完成")
            
            # 提取segment数据
            image_list, state_list, timestamp_list = extract_segment_data(
                segment_topic_data, start_sec, end_sec, fps
            )
            
            if len(image_list) == 0:
                logs.append(f"{log_prefix}: 跳过，未提取到数据")
                skipped_count += 1
                continue
            
            # 添加每一帧到数据集
            for i in range(len(image_list)):
                frame_images = image_list[i]
                state = np.asarray(state_list[i], np.float32)
                
                if action_type == 'current_state':
                    # 复制当前帧的state作为action
                    action = np.copy(state)
                elif action_type == 'next_state':
                    # 下一帧的state作为action
                    if i + 1 < len(state_list):
                        action = np.asarray(state_list[i + 1], np.float32)
                    else:
                        action = np.copy(state)
                else:
                    raise ValueError(f"未知的action_type: {action_type}")

                assert state.shape[-1] == 7, f"状态维度错误，期望7维，实际{state.shape[-1]}维"
                assert action.shape[-1] == 7, f"动作维度错误，期望7维，实际{action.shape[-1]}维"
                
                # 准备图像（至少需要两个图像，如果只有一个则复制）
                # 参考 refer_create_lerobot.py，图像尺寸应为 (640, 480, 3)
                # 注意：numpy 数组格式是 (height, width, channel)，所以 (640, 480, 3) 表示 height=640, width=480
                wrist_image_left = frame_images[0] if len(frame_images) > 0 else np.zeros((640, 480, 3), dtype=np.uint8)
                wrist_image_right = frame_images[1] if len(frame_images) > 1 else frame_images[0] if len(frame_images) > 0 else np.zeros((640, 480, 3), dtype=np.uint8)
                
                # 确保图像尺寸正确，调整为 (640, 480) 即 (height, width)
                # cv2.resize 参数是 (width, height)，所以 (480, 640) 表示 width=480, height=640
                if wrist_image_left.shape[:2] != (640, 480):
                    wrist_image_left = cv2.resize(wrist_image_left, (640, 480))
                if wrist_image_right.shape[:2] != (640, 480):
                    wrist_image_right = cv2.resize(wrist_image_right, (640, 480))
                
                # 参考 refer_create_lerobot.py，可能需要旋转图像
                # 注意：根据实际数据决定是否需要旋转，如果图像方向不对，取消下面的注释
                wrist_image_left = cv2.rotate(wrist_image_left, cv2.ROTATE_90_CLOCKWISE)
                wrist_image_right = cv2.rotate(wrist_image_right, cv2.ROTATE_90_CLOCKWISE)
                
                # 构建task字符串
                if seg_info['type'] == 'subtask':
                    task_str = '{}-{}|{}-{}'.format(
                        seg_info['taskId'], seg_info['prompt'],
                        seg_info['subtaskId'], seg_info['subtask_prompt']
                    )
                else:
                    task_str = '{}-{}'.format(seg_info['taskId'], seg_info['prompt'])
                
                dataset.add_frame({
                    "wrist_image_left": wrist_image_left,
                    "wrist_image_right": wrist_image_right,
                    "state": state,
                    "actions": action,
                    "task": task_str,  # 使用.split('|')[0].split('-')[-1]来获取task的总描述, .split('|')[-1].split('-')[-1]来获取subtask描述, 对于旧数据（仅有task）也兼容
                })
            
            dataset.save_episode()
            processed_count += 1
            logs.append(f"{log_prefix}: 成功添加 {len(image_list)} 帧")
            
        except Exception as e:
            logs.append(f"{log_prefix}: 处理失败: {e}")
            import traceback
            logs.append(traceback.format_exc())
            skipped_count += 1
    
    logs.append(f"\n完成！成功处理 {processed_count} 个segment，跳过 {skipped_count} 个segment")
    logs.append(f"数据集保存位置: {lerobot_output_path}")
    
    # 更新处理记录
    if processed_count > 0:
        try:
            update_processed_files(mcap_path, segments_json_path, output_path, repo_name, in_lerobot=True)
            logs.append(f"已更新处理记录")
        except Exception as e:
            logs.append(f"警告: 更新处理记录失败: {e}")
    
    return "\n".join(logs)


def main():
    parser = argparse.ArgumentParser(description='从mcap文件和segments.json生成lerobot数据集')
    parser.add_argument('--mcap', type=str, required=True, help='mcap文件路径')
    parser.add_argument('--segments', type=str, required=True, help='segments.json文件路径')
    parser.add_argument('--repo', type=str, required=True, help='lerobot数据集repo名称')
    parser.add_argument('--output', type=str, default=None, help='输出路径（可选）')
    parser.add_argument('--fps', type=int, default=50, help='目标帧率（默认50）')
    parser.add_argument('--clear', action='store_true', help='清空已存在的数据集')
    parser.add_argument('--action-type', type=str, default='current_state', 
                       choices=['current_state', 'next_state'],
                       help='action类型：current_state（使用当前帧state）或next_state（使用下一帧state）')
    
    args = parser.parse_args()
    
    log = create_lerobot_dataset(
        args.mcap,
        args.segments,
        args.repo,
        args.output,
        args.fps,
        args.clear,
        args.action_type
    )
    
    print(log)


if __name__ == '__main__':
    main()
