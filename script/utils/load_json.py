import json
from typing import Dict, List, Optional, Tuple
import numpy as np

def interpolate_data(data_list: List[Tuple[float, any]], target_timestamp: float) -> Optional[any]:
    """
    在数据列表中插值获取指定时间戳的数据
    
    Args:
        data_list: [(timestamp, data), ...] 按时间戳排序的列表
        target_timestamp: 目标时间戳
        
    Returns:
        插值后的数据，如果找不到则返回None
    """
    if not data_list:
        return None
    
    # 找到最接近的时间戳
    closest_idx = 0
    min_diff = abs(data_list[0][0] - target_timestamp)
    
    for i, (ts, _) in enumerate(data_list):
        diff = abs(ts - target_timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    
    # 如果时间差太大（超过0.1秒），返回None
    if min_diff > 0.1:
        return None
    
    return data_list[closest_idx][1]

def load_segments_json(json_path: str) -> List[Dict]:
    """
    读取segments.json文件，提取segment信息
    
    Args:
        json_path: segments.json文件路径
        
    Returns:
        segment列表，每个segment包含startSec, endSec, prompt
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 尝试解析为单个JSON对象
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                segments = data.get('segments', [])
                if segments:
                    return segments
                # 如果segments为空，可能数据直接在根级别
                if 'startSec' in data or 'endSec' in data:
                    return [data]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError as e:
            # 如果是 "Extra data" 错误，尝试只解析第一个完整的JSON对象
            if "Extra data" in str(e):
                try:
                    # 找到第一个完整的JSON对象（从第一个 { 到匹配的 }）
                    brace_count = 0
                    start_idx = content.find('{')
                    if start_idx >= 0:
                        for i in range(start_idx, len(content)):
                            if content[i] == '{':
                                brace_count += 1
                            elif content[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # 找到第一个完整的JSON对象
                                    first_json = content[start_idx:i+1]
                                    data = json.loads(first_json)
                                    if isinstance(data, dict):
                                        segments = data.get('segments', [])
                                        if segments:
                                            return segments
                except Exception:
                    pass
            
            # 尝试逐行解析（NDJSON格式）
            segments = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue
                try:
                    segment = json.loads(line)
                    if isinstance(segment, dict):
                        segments.append(segment)
                except json.JSONDecodeError:
                    # 跳过无法解析的行
                    continue
            
            if segments:
                return segments
            
            # 如果所有方法都失败，抛出原始错误
            error_msg = f"无法解析JSON文件: {e}"
            if hasattr(e, 'lineno') and e.lineno:
                error_msg += f", 错误位置: line {e.lineno}, column {e.colno}"
            error_msg += f". 文件路径: {json_path}"
            raise ValueError(error_msg)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"segments.json文件不存在: {json_path}")
    except Exception as e:
        raise ValueError(f"读取segments.json失败: {e}")


def extract_segment_data(
    mcap_data: Dict,
    start_sec: float,
    end_sec: float,
    fps: int = 50
) -> Tuple[List[np.ndarray], List[Dict], List[float]]:
    """
    从mcap数据中提取指定时间段的数据
    
    Args:
        mcap_data: read_mcap_file返回的数据
        start_sec: 开始时间（秒）
        end_sec: 结束时间（秒）
        fps: 目标帧率
        
    Returns:
        (images, states, timestamps) 图像列表、状态列表、时间戳列表
    """
    # 尝试找到左右图像topic
    left_topic = "/gripper/camera_fisheye_l/color/image_raw"
    right_topic = "/gripper/camera_fisheye_r/color/image_raw"
    left_end_topic = '/jbt_arm_L/current_arm_end_pose'
    right_end_topic = '/jbt_arm_R/current_arm_end_pose'
    left_gripper_topic = '/gripper/gripper_l/data'
    right_gripper_topic = '/gripper/gripper_r/data'

    images = mcap_data.get('images', {})
    pose7d_data = mcap_data.get('pose7d', {})
    
    # 确定采样时间点
    duration = end_sec - start_sec
    num_frames = int(duration * fps)
    if num_frames == 0:
        num_frames = 1
    
    timestamps = np.linspace(start_sec, end_sec, num_frames)
    
    # 找到图像topic（假设有左右两个图像topic）
    image_topics = [topic for topic in images.keys() if 'image' in topic.lower()]
    
    assert left_topic in images or right_topic in images, \
        f"未找到指定的图像topic: {left_topic} 或 {right_topic}"

    image_topics = [left_topic, right_topic]
    
    # 找到pose7d topic（用于提取状态）
    pose_topics = []
    if right_end_topic in pose7d_data:
        pose_topics.append(right_end_topic)
    elif left_end_topic in pose7d_data:
        pose_topics.append(left_end_topic)
    else:
        # 如果找不到指定的topic，使用第一个可用的
        if pose7d_data:
            pose_topics.append(list(pose7d_data.keys())[0])
    
    image_list = []
    state_list = []
    timestamp_list = []
    
    for ts in timestamps:
        # 提取图像
        frame_images = []
        for img_topic in image_topics[:2]:  # 最多取两个图像
            if img_topic in images:
                img = interpolate_data(images[img_topic], ts)
                if img is not None:
                    frame_images.append(img)
        
        # 如果图像数量不足，跳过这一帧
        if len(frame_images) < 1:
            continue
        
        # 提取状态（从pose7d数据中）
        state = np.zeros(7, dtype=np.float32)  # 默认7维状态
        for pose_topic in pose_topics:
            if pose_topic in pose7d_data:
                pose_data = interpolate_data(pose7d_data[pose_topic], ts)
                if pose_data is not None:
                    if isinstance(pose_data, np.ndarray):
                        # 直接使用numpy数组
                        if len(pose_data) >= 7:
                            state = pose_data[:7]
                        elif len(pose_data) > 0:
                            # 如果维度不足，填充0
                            state = np.pad(pose_data, (0, 7 - len(pose_data)), 'constant')[:7]
                        break
                    elif isinstance(pose_data, (list, tuple)):
                        # 转换为numpy数组
                        pose_array = np.array(pose_data, dtype=np.float32)
                        if len(pose_array) >= 7:
                            state = pose_array[:7]
                        elif len(pose_array) > 0:
                            state = np.pad(pose_array, (0, 7 - len(pose_array)), 'constant')[:7]
                        break
        
        # 确保状态是7维的numpy数组
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if len(state) != 7:
            state = np.pad(state, (0, max(0, 7 - len(state))), 'constant')[:7]
        
        image_list.append(frame_images)
        state_list.append(state)
        timestamp_list.append(ts)
    
    return image_list, state_list, timestamp_list
