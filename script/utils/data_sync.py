from utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import scipy.interpolate as si
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

def find_closest_timestamp(data_list: List[Tuple[float, Any]], target_time: float) -> Tuple[int, float]:
    """
    找到最接近目标时间的数据索引
    
    Args:
        data_list: [(timestamp, data), ...] 按时间戳排序的列表
        target_time: 目标时间戳
        
    Returns:
        (索引, 时间差)
    """
    if not data_list:
        return -1, float('inf')
    
    closest_index = 0
    closest_diff = abs(data_list[0][0] - target_time)
    
    for j, (ts, _) in enumerate(data_list):
        time_diff = abs(ts - target_time)
        if time_diff < closest_diff:
            closest_diff = time_diff
            closest_index = j
    
    return closest_index, closest_diff


def interpolate_image_data(data_list: List[Tuple[float, Any]], target_time: float) -> Optional[Any]:
    """
    对图像数据进行最近邻插值（直接取时间最近的图像）
    
    Args:
        data_list: [(timestamp, image), ...] 按时间戳排序的列表
        target_time: 目标时间戳
        
    Returns:
        插值后的图像，如果列表为空返回None
    """
    if not data_list:
        return None
    
    closest_idx, _ = find_closest_timestamp(data_list, target_time)
    return data_list[closest_idx][1]


def interpolate_other_data(data_list: List[Tuple[float, Any]], target_time: float, 
                           data_type: str) -> Optional[Any]:
    """
    对其他类型数据进行插值
    
    Args:
        data_list: [(timestamp, data), ...] 按时间戳排序的列表
        target_time: 目标时间戳
        data_type: 数据类型（'pose7d' 或其他）
        
    Returns:
        插值后的数据，如果列表为空返回None
    """
    if not data_list:
        return None
    
    if len(data_list) == 1:
        # 只有一个数据点，直接返回
        return data_list[0][1]
    
    times = np.array([ts for ts, _ in data_list])
    values = np.array([val for _, val in data_list])
    
    # 检查数据维度
    if len(values.shape) == 1:
        # 一维数据，使用线性插值
        interp_func = si.interp1d(
            times, values, 
            axis=0, 
            kind='linear',
            bounds_error=False,
            fill_value=(values[0], values[-1])
        )
        return interp_func(target_time)
    elif len(values.shape) == 2:
        # 多维数据
        if data_type == 'pose7d' and values.shape[1] >= 6:
            # 位姿数据，使用PoseTrajectoryInterpolator
            try:
                pose_interp = PoseTrajectoryInterpolator(times, values)
                result = pose_interp(target_time)
                # 确保输出是1D数组
                if len(result.shape) == 1:
                    return result
                else:
                    return result[0] if result.shape[0] == 1 else result
            except Exception as e:
                print(f"警告: 位姿插值失败，使用线性插值: {e}")
                # 降级到线性插值
                interp_func = si.interp1d(
                    times, values,
                    axis=0,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(values[0], values[-1])
                )
                result = interp_func(target_time)
                return result[0] if len(result.shape) > 1 and result.shape[0] == 1 else result
        else:
            # 其他多维数据，使用线性插值
            interp_func = si.interp1d(
                times, values,
                axis=0,
                kind='linear',
                bounds_error=False,
                fill_value=(values[0], values[-1])
            )
            result = interp_func(target_time)
            return result[0] if len(result.shape) > 1 and result.shape[0] == 1 else result
    else:
        # 高维数据，使用最近邻
        closest_idx, _ = find_closest_timestamp(data_list, target_time)
        return data_list[closest_idx][1]


def sync_topic_data(topic_data: Dict[str, Dict[str, List[Tuple[float, Any]]]], 
                     time_diff_limit: float = 0.003) -> Dict[str, Dict[str, List[Tuple[float, Any]]]]:
    """
    同步所有topic数据，以右手图像为主时间轴
    
    策略：
    1. 以右手图像的时间戳为主时间轴
    2. 左手图像：如果在time_diff_limit时间内有数据就用，否则用最近邻插值
    3. 其他topic：如果在time_diff_limit时间内有数据就用，否则插值
    
    Args:
        topic_data: 从load_mcap_data返回的数据
        time_diff_limit: 时间差限制（秒）
        
    Returns:
        同步后的数据，格式相同
    """
    # 找到右手图像topic（主时间轴）
    right_image_topic = "/gripper/camera_fisheye_r/color/image_raw"
    left_image_topic = "/gripper/camera_fisheye_l/color/image_raw"
    
    # 获取右手图像的时间戳列表作为主时间轴
    if 'images' not in topic_data or right_image_topic not in topic_data['images']:
        print("警告: 未找到右手图像topic，使用所有时间戳的并集")
        # 回退到原来的逻辑：收集所有时间戳
        all_timestamps = []
        for data_type, topics_data in topic_data.items():
            for topic_name, data_list in topics_data.items():
                for ts, _ in data_list:
                    all_timestamps.append(ts)
        master_timestamps = sorted(set(all_timestamps))
    else:
        right_image_data = topic_data['images'][right_image_topic]
        master_timestamps = [ts for ts, _ in right_image_data]
        print(f"以右手图像为主时间轴，共 {len(master_timestamps)} 个时间戳")
    
    if not master_timestamps:
        return topic_data
    
    # 初始化同步后的数据
    synced_data = {}
    for data_type in topic_data.keys():
        synced_data[data_type] = {}
        for topic_name in topic_data[data_type].keys():
            synced_data[data_type][topic_name] = []
    
    # 对每个主时间戳进行同步
    for frame_time in master_timestamps:
        # 处理右手图像（主时间轴，直接使用对应时间戳的数据）
        if 'images' in topic_data and right_image_topic in topic_data['images']:
            right_data_list = topic_data['images'][right_image_topic]
            # 找到最接近的数据（应该就是当前时间戳的数据，因为master_timestamps来自右手图像）
            closest_idx, closest_diff = find_closest_timestamp(right_data_list, frame_time)
            # 右手图像是主时间轴，应该总是有对应的数据
            synced_data['images'][right_image_topic].append(right_data_list[closest_idx])
        
        # 处理左手图像（如果在time_diff_limit内有数据就用，否则插值）
        if 'images' in topic_data and left_image_topic in topic_data['images']:
            left_data_list = topic_data['images'][left_image_topic]
            closest_idx, closest_diff = find_closest_timestamp(left_data_list, frame_time)
            if closest_diff <= time_diff_limit:
                # 在时间差限制内，直接使用
                synced_data['images'][left_image_topic].append(left_data_list[closest_idx])
            else:
                # 超出时间差限制，使用最近邻插值
                interpolated_image = interpolate_image_data(left_data_list, frame_time)
                if interpolated_image is not None:
                    synced_data['images'][left_image_topic].append((frame_time, interpolated_image))
        
        # 处理其他topic
        for data_type, topics_data in topic_data.items():
            if data_type == 'images':
                continue  # 图像已经处理过了
            
            for topic_name, data_list in topics_data.items():
                if not data_list:
                    continue
                
                closest_idx, closest_diff = find_closest_timestamp(data_list, frame_time)
                
                if closest_diff <= time_diff_limit:
                    # 在时间差限制内，直接使用
                    synced_data[data_type][topic_name].append(data_list[closest_idx])
                else:
                    # 超出时间差限制，进行插值
                    if data_type == 'pose7d':
                        # 位姿数据使用位姿插值
                        interpolated_value = interpolate_other_data(data_list, frame_time, 'pose7d')
                    else:
                        # 其他数据使用线性插值
                        interpolated_value = interpolate_other_data(data_list, frame_time, data_type)
                    
                    if interpolated_value is not None:
                        synced_data[data_type][topic_name].append((frame_time, interpolated_value))
    
    print(f"数据同步完成: 主时间轴 {len(master_timestamps)} 个时间戳")
    return synced_data
