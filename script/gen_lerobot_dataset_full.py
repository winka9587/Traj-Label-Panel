# 不识别subtask，将整个task视为一个任务
# 使用mcap文件和标签文件json（segments.json），生成lerobot数据集
# 1. 读取mcap文件，提取画面和topic数据
# 2. 读取每个mcap文件对应的标签文件json，提取整个mcap轨迹对应的多个segment的start和end时间戳
# 3. 对于有subtasks的segment，使用第一个子任务的开始时间和最后一个子任务的结束时间
# 4. 对每个segment，截取对应的画面和topic数据，保存为lerobot数据集格式

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import cv2
import hashlib
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

import sys, os
logger.info(f"[DEBUG] __file__={__file__}")
logger.info(f"[BOOT] cwd={os.getcwd()}")
logger.info(f"[BOOT] python={sys.executable}")
logger.info("[BOOT] logging is alive")


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
    计算文件的快速哈希值（用于文件去重）
    对于大文件，只读取部分内容以加快速度
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件的十六进制哈希值
    """
    import os
    file_size = os.path.getsize(file_path)
    file_stat = os.stat(file_path)
    
    # 对于小文件（<10MB），使用完整文件哈希
    # 对于大文件，使用文件大小 + 修改时间 + 部分内容
    if file_size < 10 * 1024 * 1024:  # 10MB
        # 小文件：使用 MD5（比 SHA256 快）
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
        return md5_hash.hexdigest()
    else:
        # 大文件：使用文件大小 + 修改时间 + 文件头尾部分内容
        # 读取前 1MB + 后 1MB + 文件大小 + 修改时间
        sample_size = 1024 * 1024  # 1MB
        md5_hash = hashlib.md5()
        
        # 添加文件大小和修改时间
        md5_hash.update(str(file_size).encode())
        md5_hash.update(str(file_stat.st_mtime).encode())
        
        with open(file_path, "rb") as f:
            # 读取文件开头
            head_data = f.read(sample_size)
            md5_hash.update(head_data)
            
            # 读取文件结尾
            if file_size > sample_size * 2:
                f.seek(file_size - sample_size)
                tail_data = f.read(sample_size)
                md5_hash.update(tail_data)
        
        return md5_hash.hexdigest()


def get_processed_files_path(output_path: Optional[str], repo_name: str) -> Path:
    """
    获取处理记录JSON文件的路径（位于lerobot数据目录下）
    
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
    
    # 将文件放在lerobot数据目录下
    return base_path / repo_name / "processed_files.json"


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
        logger.warning(f"无法读取处理记录文件 {processed_files_path}: {e}")
        return {}


def save_processed_files(
    output_path: Optional[str],
    repo_name: str,
    processed_files: Dict[str, Any]
) -> None:
    """
    保存已处理文件记录（使用原子性写入）
    
    Args:
        output_path: 输出路径（可选）
        repo_name: lerobot数据集repo名称
        processed_files: 已处理文件记录的字典
    """
    processed_files_path = get_processed_files_path(output_path, repo_name)
    
    # 确保目录存在
    processed_files_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用临时文件+原子重命名确保原子性写入
    temp_path = processed_files_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # 确保数据写入磁盘
        
        # 原子性重命名（在支持的系统上这是原子操作）
        temp_path.replace(processed_files_path)
    except Exception as e:
        logger.warning(f"无法保存处理记录文件 {processed_files_path}: {e}")
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise


def check_already_processed(
    mcap_path: str,
    segments_json_path: str,
    segment_idx: int,
    output_path: Optional[str],
    repo_name: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    检查特定的segment是否已经处理过
    
    Args:
        mcap_path: mcap文件路径
        segments_json_path: segments.json文件路径
        segment_idx: segment索引
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
        logger.warning(f"无法计算文件哈希: {e}")
        return False, None
    
    # 组合哈希作为唯一标识（包含segment索引）
    combined_hash = f"{mcap_hash}_{segments_hash}_{segment_idx}"
    
    # 加载处理记录
    processed_files = load_processed_files(output_path, repo_name)
    
    # 检查是否已处理
    if combined_hash in processed_files:
        record = processed_files[combined_hash]
        logger.debug(f"找到已处理记录: segment {segment_idx}, {record}")
        return True, record
    
    return False, None


def update_processed_files(
    mcap_path: str,
    segments_json_path: str,
    segment_idx: int,
    output_path: Optional[str],
    repo_name: str,
    in_lerobot: bool = True
) -> None:
    """
    更新已处理文件记录（针对特定segment）
    
    Args:
        mcap_path: mcap文件路径
        segments_json_path: segments.json文件路径
        segment_idx: segment索引
        output_path: 输出路径（可选）
        repo_name: lerobot数据集repo名称
        in_lerobot: 是否已经在lerobot数据集中
    """
    # 计算文件哈希
    try:
        mcap_hash = calculate_file_hash(mcap_path)
        segments_hash = calculate_file_hash(segments_json_path)
    except Exception as e:
        logger.warning(f"无法计算文件哈希: {e}")
        raise
    
    # 组合哈希作为唯一标识（包含segment索引）
    combined_hash = f"{mcap_hash}_{segments_hash}_{segment_idx}"
    
    # 加载现有记录
    processed_files = load_processed_files(output_path, repo_name)
    
    # 更新记录
    processed_files[combined_hash] = {
        "mcap_path": mcap_path,
        "segments_json_path": segments_json_path,
        "segment_idx": segment_idx,
        "mcap_hash": mcap_hash,
        "segments_hash": segments_hash,
        "repo_name": repo_name,
        "in_lerobot": in_lerobot,
        "processed_time": datetime.now().isoformat()  # 使用当前时间作为处理时间戳
    }
    
    # 保存记录（使用原子性写入）
    save_processed_files(output_path, repo_name, processed_files)


def print_topic_data_summary(topic_data: Dict[str, Dict[str, List[Tuple[float, Any]]]]):
    for data_type, topics_data in topic_data.items():
        for topic_name, data_list in topics_data.items():
            if len(data_list) > 1:
                start_time = data_list[0][0]
                end_time = data_list[-1][0]
                time_span = end_time - start_time
                fps = len(data_list) / time_span if time_span > 0 else 0.0
                logger.debug(f"已加载 {data_type} topic '{topic_name}'，数据量: {len(data_list)}，时间范围: {time_span:.2f}s，帧率: {fps:.2f} Hz")
            else:
                logger.debug(f"已加载 {data_type} topic '{topic_name}'，数据量: {len(data_list)}")


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
    # 检查文件是否存在
    if not os.path.isfile(mcap_path):
        error_msg = f"错误: mcap文件不存在: {mcap_path}"
        logger.error(error_msg)
        return error_msg
    
    if not os.path.isfile(segments_json_path):
        error_msg = f"错误: segments.json文件不存在: {segments_json_path}"
        logger.error(error_msg)
        return error_msg
    
    logger.info(f"读取mcap文件: {mcap_path}")
    logger.info(f"读取segments文件: {segments_json_path}")
    
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
        logger.info(f"成功读取segments.json，找到 {len(segments)} 个segment")
    except Exception as e:
        error_msg = f"读取segments.json失败: {e}"
        logger.error(error_msg)
        return error_msg
    
    # 收集segment信息（用于后续处理）
    segment_info_list = []  # 存储每个segment的信息（不识别subtask）
    
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
            segment_info_list.append({
                'type': 'segment',
                'segment_idx': idx,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'prompt': prompt,
                'taskId': taskId,
            })
        else:
            # 有subtasks，不识别subtask，将整个task视为一个任务
            # 使用第一个子任务的开始时间和最后一个子任务的结束时间
            subtask_start_times = []
            subtask_end_times = []
            
            for subtask_meta in subtasks:
                subtask_start_sec = subtask_meta.get('startSec', None)
                subtask_end_sec = subtask_meta.get('endSec', None)
                if subtask_start_sec is None or subtask_end_sec is None:
                    continue
                if subtask_end_sec <= subtask_start_sec:
                    continue
                
                subtask_start_times.append(subtask_start_sec)
                subtask_end_times.append(subtask_end_sec)
            
            if len(subtask_start_times) > 0:
                # 使用第一个子任务的开始时间和最后一个子任务的结束时间
                task_start_sec = min(subtask_start_times)
                task_end_sec = max(subtask_end_times)
                
                segment_info_list.append({
                    'type': 'segment',
                    'segment_idx': idx,
                    'start_sec': task_start_sec,
                    'end_sec': task_end_sec,
                    'prompt': prompt,
                    'taskId': taskId,
                })
    
    if not segment_info_list:
        error_msg = "错误: 没有有效的时间段"
        logger.error(error_msg)
        return error_msg
    
    logger.info(f"收集到 {len(segment_info_list)} 个有效的时间段")
    
    # 一次性加载所有需要的数据（load_mcap_data会根据segments自动计算时间范围，并添加3秒冗余）
    try:
        logger.info("从mcap文件一次性加载数据（仅加载segments时间范围内的数据，前后冗余3秒）...")
        all_topic_data = load_mcap_data(mcap_path, topic_list, 
                                       segments=segments, 
                                       buffer_seconds=3.0)
        logger.info("数据加载完成")
    except Exception as e:
        error_msg = f"读取mcap文件失败: {e}"
        logger.error(error_msg)
        return error_msg
    
    # 确定输出路径
    if output_path is None:
        lerobot_output_path = HF_LEROBOT_HOME / repo_name
    else:
        lerobot_output_path = Path(output_path) / repo_name
    
    # 清空已存在的数据集
    if clear_dataset and lerobot_output_path.exists():
        import shutil
        shutil.rmtree(lerobot_output_path)
        logger.info(f"已清空已存在的数据集: {lerobot_output_path}")
        
        # 如果清空数据集，也清除相关的处理记录（删除所有相关的segment记录）
        try:
            # 计算文件哈希
            mcap_hash = calculate_file_hash(mcap_path)
            segments_hash = calculate_file_hash(segments_json_path)
            
            # 加载并删除所有相关的segment记录
            processed_files = load_processed_files(output_path, repo_name)
            keys_to_delete = [
                key for key in processed_files.keys() 
                if key.startswith(f"{mcap_hash}_{segments_hash}_")
            ]
            if keys_to_delete:
                for key in keys_to_delete:
                    del processed_files[key]
                save_processed_files(output_path, repo_name, processed_files)
                logger.info(f"已清除 {len(keys_to_delete)} 条相关的处理记录")
        except Exception as e:
            logger.warning(f"清除处理记录失败: {e}")
    
    # 创建或加载lerobot数据集
    lerobot_exists = lerobot_output_path.exists()
    if lerobot_exists:
        logger.info(f"数据集已存在，追加新episode: {lerobot_output_path}")
        dataset = LeRobotDataset(repo_id=repo_name)
        dataset.start_image_writer(
            num_processes=5,
            num_threads=10,
        )
    else:
        logger.info(f"创建新数据集: {lerobot_output_path}")
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
    
    # 处理每个segment（不识别subtask，将整个task视为一个任务）
    processed_count = 0
    skipped_count = 0
    
    for seg_info in segment_info_list:
        start_sec = seg_info['start_sec']
        end_sec = seg_info['end_sec']
        segment_idx = seg_info['segment_idx']
        
        log_prefix = f"Segment {segment_idx}"
        
        # 检查该segment是否已经处理过
        already_processed, record = check_already_processed(
            mcap_path, segments_json_path, segment_idx, output_path, repo_name
        )
        if already_processed and record:
            if record.get('in_lerobot', False):
                logger.info(f"{log_prefix}: 已处理过，跳过")
                skipped_count += 1
                continue
            else:
                logger.info(f"{log_prefix}: 已处理过但未在lerobot中，重新处理...")
        
        logger.info(f"处理{log_prefix}: {start_sec:.2f}s - {end_sec:.2f}s")
        
        try:
            # 从已加载的数据中裁剪当前segment的时间范围
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
            logger.info(f"{log_prefix}: 开始数据同步&插值...")
            segment_topic_data = sync_topic_data(segment_topic_data, time_diff_limit=0.03)
            print_topic_data_summary(segment_topic_data)
            logger.info(f"{log_prefix}: 数据同步&插值完成")
            
            # 提取segment数据
            image_list, state_list, timestamp_list = extract_segment_data(
                segment_topic_data, start_sec, end_sec, fps
            )
            
            if len(image_list) == 0:
                logger.warning(f"{log_prefix}: 跳过，未提取到数据")
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
                
                # 构建task字符串（不识别subtask，只使用task信息）
                task_str = '{}-{}'.format(seg_info['taskId'], seg_info['prompt'])
                
                dataset.add_frame({
                    "wrist_image_left": wrist_image_left,
                    "wrist_image_right": wrist_image_right,
                    "state": state,
                    "actions": action,
                    "task": task_str,  # 格式为 '{taskId}-{prompt}'，不包含subtask信息
                })
            
            # 保存episode
            dataset.save_episode()
            
            # 立即更新处理记录（确保原子性：save_episode成功后才更新记录）
            try:
                update_processed_files(
                    mcap_path, segments_json_path, segment_idx, 
                    output_path, repo_name, in_lerobot=True
                )
                logger.debug(f"{log_prefix}: 已更新处理记录")
            except Exception as e:
                # 如果更新记录失败，记录警告但不影响主流程
                # 因为save_episode已经成功，数据已经保存
                logger.warning(f"{log_prefix}: 更新处理记录失败: {e}")
            
            processed_count += 1
            logger.info(f"{log_prefix}: 成功添加 {len(image_list)} 帧")
            
        except Exception as e:
            logger.error(f"{log_prefix}: 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            skipped_count += 1
            # save_episode失败，不更新处理记录（保持原子性）
    
    logger.info(f"完成！成功处理 {processed_count} 个segment，跳过 {skipped_count} 个segment")
    logger.info(f"mcap文件: {mcap_path}")
    logger.info(f"segments文件: {segments_json_path}")
    logger.info(f"数据集保存位置: {lerobot_output_path}")
    
    # 注意：处理记录已在每个segment的save_episode后立即更新，这里不再需要整体更新
    
    # 显示数据集统计信息
    try:
        if lerobot_output_path.exists():
            # 重新加载数据集以获取统计信息
            dataset_for_info = LeRobotDataset(repo_id=repo_name)
            total_frames = len(dataset_for_info)
            
            # 尝试获取episode数量
            try:
                # LeRobotDataset可能有num_episodes属性，或者可以通过其他方式获取
                if hasattr(dataset_for_info, 'num_episodes'):
                    num_episodes = dataset_for_info.num_episodes
                elif hasattr(dataset_for_info, 'info') and hasattr(dataset_for_info.info, 'num_episodes'):
                    num_episodes = dataset_for_info.info.num_episodes
                else:
                    # 如果没有直接属性，尝试从数据目录统计episode目录数量
                    episode_dirs = [d for d in lerobot_output_path.iterdir() 
                                  if d.is_dir() and d.name.startswith('episode')]
                    num_episodes = len(episode_dirs)
                
                logger.info("数据集统计信息:")
                logger.info(f"  总帧数: {total_frames}")
                logger.info(f"  Episode数量: {num_episodes}")
            except Exception as e:
                # 如果获取episode数量失败，至少显示总帧数
                logger.info("数据集统计信息:")
                logger.info(f"  总帧数: {total_frames}")
                logger.warning(f"  无法获取episode数量: {e}")
        else:
            logger.info("数据集统计信息: 数据集目录不存在")
    except Exception as e:
        logger.warning(f"无法获取数据集统计信息: {e}")
    
    return f"处理完成：成功 {processed_count} 个，跳过 {skipped_count} 个"


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
    
    result = create_lerobot_dataset(
        args.mcap,
        args.segments,
        args.repo,
        args.output,
        args.fps,
        args.clear,
        args.action_type
    )
    
    # 结果已通过 logger 输出，这里只返回状态
    if result:
        logger.info(f"最终结果: {result}")


if __name__ == '__main__':
    main()
