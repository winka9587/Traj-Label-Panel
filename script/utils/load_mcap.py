
import pathlib
import numpy as np
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Dict

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: cv2库未安装，图像解码功能可能不可用")

# 添加项目路径以导入openpi模块
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from openpi.training.config import get_config
from openpi import transforms as _transforms
from openpi.policies import pika_policy

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 如果没有tqdm，创建一个简单的占位符
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from mcap.reader import make_reader
    try:
        from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory
    except ImportError:
        from mcap_ros2.decoder import Ros2DecoderFactory
    MCAP_AVAILABLE = True
except ImportError:
    MCAP_AVAILABLE = False
    print("警告: mcap库未安装，请安装: pip install mcap mcap-ros2-support")
def decode_image_message(message: Any) -> Optional[np.ndarray]:
    """
    解码ROS2图像消息为numpy数组
    
    Args:
        message: ROS2图像消息对象
        
    Returns:
        解码后的图像数组，失败返回None
    """
    if not CV2_AVAILABLE:
        return None
    
    try:
        # 获取图像数据
        if hasattr(message, 'data'):
            img_data = message.data
        elif hasattr(message, 'image_data'):
            img_data = message.image_data
        else:
            return None
        
        if not isinstance(img_data, (bytes, bytearray, np.ndarray)):
            return None
        
        img_bytes = np.frombuffer(img_data, dtype=np.uint8)
        
        # 获取编码格式
        encoding = getattr(message, 'encoding', 'bgr8')
        
        # 获取图像尺寸
        width = getattr(message, 'width', None)
        height = getattr(message, 'height', None)
        
        # 如果已知尺寸，直接reshape
        if width is not None and height is not None:
            if encoding in ['rgb8', 'bgr8', 'rgba8', 'bgra8']:
                channels = 3 if encoding in ['rgb8', 'bgr8'] else 4
                expected_size = width * height * channels
                if len(img_bytes) == expected_size:
                    img = img_bytes.reshape((height, width, channels))
                    if encoding == 'rgb8':
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif encoding == 'rgba8':
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    elif encoding == 'bgra8':
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    return img
        
        # 尝试使用cv2解码（适用于压缩格式如jpeg、png）
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        
        return None
    except Exception as e:
        print(f"解码图像失败: {e}")
        return None


def extract_message_data(message: Any) -> Dict[str, Any]:
    """
    从ROS2消息中提取数据为字典
    
    Args:
        message: ROS2消息对象
        
    Returns:
        包含消息字段的字典
    """
    msg_data = {}
    try:
        x, y, z = 0.0, 0.0, 0.0
        roll, pitch, yaw = 0.0, 0.0, 0.0
        gripper = 0.0
        
        # 方法1: 从PoseStamped格式提取 (geometry_msgs/PoseStamped)
        if hasattr(ros_msg, 'pose') and hasattr(ros_msg.pose, 'position') and hasattr(ros_msg.pose, 'orientation'):
            x = ros_msg.pose.position.x
            y = ros_msg.pose.position.y
            z = ros_msg.pose.position.z
            qx = ros_msg.pose.orientation.x
            qy = ros_msg.pose.orientation.y
            qz = ros_msg.pose.orientation.z
            qw = ros_msg.pose.orientation.w
            roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        # 方法2: 直接从Pose格式提取 (geometry_msgs/Pose)
        elif hasattr(ros_msg, 'position') and hasattr(ros_msg, 'orientation'):
            x = ros_msg.position.x
            y = ros_msg.position.y
            z = ros_msg.position.z
            qx = ros_msg.orientation.x
            qy = ros_msg.orientation.y
            qz = ros_msg.orientation.z
            qw = ros_msg.orientation.w
            roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        else:
            # 方法3: 从字典格式提取（如果已经是提取后的数据）
            msg_data = extract_message_data(ros_msg)
            if 'pose' in msg_data:
                pose = msg_data['pose']
                if isinstance(pose, dict):
                    if 'position' in pose:
                        pos = pose['position']
                        if isinstance(pos, dict):
                            x = pos.get('x', 0.0)
                            y = pos.get('y', 0.0)
                            z = pos.get('z', 0.0)
                    if 'orientation' in pose:
                        orient = pose['orientation']
                        if isinstance(orient, dict):
                            qx = orient.get('x', 0.0)
                            qy = orient.get('y', 0.0)
                            qz = orient.get('z', 0.0)
                            qw = orient.get('w', 1.0)
                            roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
            elif 'position' in msg_data and 'orientation' in msg_data:
                pos = msg_data['position']
                if isinstance(pos, dict):
                    x = pos.get('x', 0.0)
                    y = pos.get('y', 0.0)
                    z = pos.get('z', 0.0)
                orient = msg_data['orientation']
                if isinstance(orient, dict):
                    qx = orient.get('x', 0.0)
                    qy = orient.get('y', 0.0)
                    qz = orient.get('z', 0.0)
                    qw = orient.get('w', 1.0)
                    roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
            elif 'x' in msg_data and 'y' in msg_data and 'z' in msg_data:
                x = msg_data.get('x', 0.0)
                y = msg_data.get('y', 0.0)
                z = msg_data.get('z', 0.0)
                if 'roll' in msg_data and 'pitch' in msg_data and 'yaw' in msg_data:
                    roll = msg_data.get('roll', 0.0)
                    pitch = msg_data.get('pitch', 0.0)
                    yaw = msg_data.get('yaw', 0.0)
                elif 'qx' in msg_data or 'orientation_x' in msg_data:
                    qx = msg_data.get('qx', msg_data.get('orientation_x', 0.0))
                    qy = msg_data.get('qy', msg_data.get('orientation_y', 0.0))
                    qz = msg_data.get('qz', msg_data.get('orientation_z', 0.0))
                    qw = msg_data.get('qw', msg_data.get('orientation_w', 1.0))
                    roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        
        # 提取gripper状态
        if hasattr(ros_msg, 'gripper'):
            gripper = float(ros_msg.gripper)
        elif hasattr(ros_msg, 'gripper_state'):
            gripper = float(ros_msg.gripper_state)
        elif hasattr(ros_msg, 'gripper_position'):
            gripper = float(ros_msg.gripper_position)
        else:
            # 尝试从字典中提取
            msg_data = extract_message_data(ros_msg) if not isinstance(ros_msg, dict) else ros_msg
            if 'gripper' in msg_data:
                gripper = float(msg_data['gripper'])
            elif 'gripper_state' in msg_data:
                gripper = float(msg_data['gripper_state'])
            elif 'gripper_position' in msg_data:
                gripper = float(msg_data['gripper_position'])
        
        return np.array([x, y, z, roll, pitch, yaw, gripper], dtype=np.float32)
    except Exception as e:
        return None


def extract_joint_states_from_message(ros_msg: Any) -> Optional[np.ndarray]:
    """
    从单个ROS2消息中提取joint_states数据 (6维position + 6维velocity)
    
    Args:
        ros_msg: ROS2消息对象
        
    Returns:
        12维numpy数组 [position_6d, velocity_6d]，失败返回None
    """
    try:
        position = np.zeros(6, dtype=np.float32)
        velocity = np.zeros(6, dtype=np.float32)
        
        # 方法1: 从标准的JointState消息结构提取 (sensor_msgs/JointState)
        if hasattr(ros_msg, 'position') and hasattr(ros_msg, 'velocity'):
            pos_data = ros_msg.position
            vel_data = ros_msg.velocity
            
            if isinstance(pos_data, (list, tuple, np.ndarray)):
                pos_array = np.array(pos_data, dtype=np.float32)
                position = pos_array[:6] if len(pos_array) >= 6 else np.pad(pos_array, (0, 6 - len(pos_array)), 'constant')[:6]
            
            if isinstance(vel_data, (list, tuple, np.ndarray)):
                vel_array = np.array(vel_data, dtype=np.float32)
                velocity = vel_array[:6] if len(vel_array) >= 6 else np.pad(vel_array, (0, 6 - len(vel_array)), 'constant')[:6]
        else:
            # 方法2: 从字典格式提取
            msg_data = extract_message_data(ros_msg) if not isinstance(ros_msg, dict) else ros_msg
            
            if 'position' in msg_data:
                pos_data = msg_data['position']
                if isinstance(pos_data, (list, np.ndarray)):
                    pos_array = np.array(pos_data, dtype=np.float32)
                    position = pos_array[:6] if len(pos_array) >= 6 else np.pad(pos_array, (0, 6 - len(pos_array)), 'constant')[:6]
            
            if 'velocity' in msg_data:
                vel_data = msg_data['velocity']
                if isinstance(vel_data, (list, np.ndarray)):
                    vel_array = np.array(vel_data, dtype=np.float32)
                    velocity = vel_array[:6] if len(vel_array) >= 6 else np.pad(vel_array, (0, 6 - len(vel_array)), 'constant')[:6]
            
            # 尝试其他可能的字段名
            if np.all(position == 0):
                for key in ['positions', 'joint_positions', 'pos', 'joint_pos']:
                    if key in msg_data:
                        pos_data = msg_data[key]
                        if isinstance(pos_data, (list, np.ndarray)):
                            pos_array = np.array(pos_data, dtype=np.float32)
                            position = pos_array[:6] if len(pos_array) >= 6 else np.pad(pos_array, (0, 6 - len(pos_array)), 'constant')[:6]
                            break
            
            if np.all(velocity == 0):
                for key in ['velocities', 'joint_velocities', 'vel', 'joint_vel']:
                    if key in msg_data:
                        vel_data = msg_data[key]
                        if isinstance(vel_data, (list, np.ndarray)):
                            vel_array = np.array(vel_data, dtype=np.float32)
                            velocity = vel_array[:6] if len(vel_array) >= 6 else np.pad(vel_array, (0, 6 - len(vel_array)), 'constant')[:6]
                            break
        
        # 合并为12维向量 [position_6d, velocity_6d]
        joint_state = np.concatenate([position, velocity], axis=0)
        return joint_state
    except Exception as e:
        return None


from typing import Dict, List, Optional, Any, Tuple

def get_mcap_time_range(mcap_path: str, use_sampling: bool = False, sample_size: int = 1000) -> Tuple[float, float]:
    """
    获取mcap文件的时间范围（最小和最大时间戳）
    
    优化版本：仅读取时间戳元数据（不解码消息内容），大幅提升速度
    
    Args:
        mcap_path: mcap文件路径
        use_sampling: 是否使用采样方法（只读取前N个和后N个消息），默认False（完整遍历但不解码）
        sample_size: 采样大小，每端读取的消息数量，默认1000（仅在use_sampling=True时使用）
        
    Returns:
        (min_time, max_time): 最小和最大时间戳（秒）
    """
    if not MCAP_AVAILABLE:
        raise ImportError("mcap库未安装，请安装: pip install mcap mcap-ros2-support")
    
    min_time = float('inf')
    max_time = float('-inf')
    
    with open(mcap_path, "rb") as f:
        # 不使用decoder factory，因为我们只需要时间戳，不需要解码消息内容
        reader = make_reader(f)
        
        if use_sampling:
            # 采样方法：只读取前N个和后N个消息的时间戳（更快但不一定准确）
            timestamps = []
            
            for schema, channel, message in reader.iter_messages():
                timestamp_ns = message.log_time if hasattr(message, 'log_time') else 0
                timestamp_sec = timestamp_ns / 1e9
                timestamps.append(timestamp_sec)
            
            if len(timestamps) == 0:
                return (0.0, 0.0)
            
            # 如果消息数量少于采样大小，使用所有时间戳
            if len(timestamps) <= sample_size * 2:
                min_time = min(timestamps)
                max_time = max(timestamps)
            else:
                # 取前N个和后N个时间戳
                front_timestamps = timestamps[:sample_size]
                back_timestamps = timestamps[-sample_size:]
                all_sampled = front_timestamps + back_timestamps
                min_time = min(all_sampled)
                max_time = max(all_sampled)
        else:
            # 完整遍历但不解码（快速且准确）
            # 只读取时间戳元数据，不解码消息内容，速度比iter_decoded_messages快很多
            for schema, channel, message in reader.iter_messages():
                timestamp_ns = message.log_time if hasattr(message, 'log_time') else 0
                timestamp_sec = timestamp_ns / 1e9
                
                if timestamp_sec < min_time:
                    min_time = timestamp_sec
                if timestamp_sec > max_time:
                    max_time = timestamp_sec
    
    if min_time == float('inf'):
        return (0.0, 0.0)
    
    return (min_time, max_time)


def create_time_progress_info(current_time: float,
                              start_time: Optional[float] = None, end_time: Optional[float] = None,
                              time_ranges: Optional[List[Tuple[float, float]]] = None) -> str:
    """
    创建时间信息字符串（简化版，不需要文件时间范围）
    
    Args:
        current_time: 当前处理的时间戳（秒）
        start_time: 开始时间（秒），None表示不限制（用于单个时间范围）
        end_time: 结束时间（秒），None表示不限制（用于单个时间范围）
        time_ranges: 多个时间范围列表，格式为 [(start1, end1), (start2, end2), ...]（优先级高于start_time/end_time）
        
    Returns:
        时间信息字符串
    """
    time_info = f"时间: {current_time:.1f}s"
    if current_time < start_time:
        time_info = 'skip'
    elif current_time > end_time:
        time_info = 'skip'
    else:
        progress_rate = (current_time - start_time)/(end_time - start_time)
        time_info = f"时间进度: {progress_rate:.1%}"
    return time_info


def read_mcap_file(mcap_path: str, show_progress: bool = True, topic_list: Optional[Dict[str, List[str]]] = None, 
                   start_time: Optional[float] = None, end_time: Optional[float] = None,
                   time_ranges: Optional[List[Tuple[float, float]]] = None) -> Dict:
    """
    读取mcap文件，提取图像和topic数据。
    如果提供了topic_list，会根据topic的类型直接提取相应的数据：
    - pose7d: 提取为7维向量 [x, y, z, roll, pitch, yaw, gripper]
    - joint_states: 提取为12维向量 [6维position + 6维velocity]
    - images: 直接存储图像数据
    
    Args:
        mcap_path: mcap文件路径
        show_progress: 是否显示进度条
        topic_list: 可选的字典，key是数据类型（"pose7d", "joint_states", "images"），
                   value是topic名称列表。如果提供，将直接提取相应类型的数据。
                   例如:
                   {
                       "pose7d": ["/jbt_arm_R/current_arm_end_pose", ...],
                       "joint_states": ["/jbt_arm_R/current_arm_joint_state", ...],
                       "images": ["/gripper/camera_fisheye_l/color/image_raw", ...]
                   }
        start_time: 可选，开始时间（秒），只加载此时间之后的数据（当time_ranges为None时使用）
        end_time: 可选，结束时间（秒），只加载此时间之前的数据（当time_ranges为None时使用）
        time_ranges: 可选，多个时间范围列表，格式为 [(start1, end1), (start2, end2), ...]。
                     如果提供，只加载这些时间范围内的数据。优先级高于start_time/end_time
        
    Returns:
        如果提供了topic_list，返回格式为:
        {
            "pose7d": {topic_name: [(timestamp, np.array([x, y, z, roll, pitch, yaw, gripper])), ...]},
            "joint_states": {topic_name: [(timestamp, np.array([pos_6d, vel_6d])), ...]},
            "images": {topic_name: [(timestamp, image), ...]}
        }
        否则返回:
        {
            'images': {topic_name: [(timestamp, image_data), ...]},
            'topics': {topic_name: [(timestamp, message_data), ...]}
        }
    """
    if not MCAP_AVAILABLE:
        raise ImportError("mcap库未安装，请安装: pip install mcap mcap-ros2-support")
    
    # 如果提供了topic_list，构建topic到类型的映射
    topic_to_type = {}
    if topic_list is not None:
        for data_type, topics in topic_list.items():
            for topic_name in topics:
                topic_to_type[topic_name] = data_type
    
    # 根据是否提供topic_list，使用不同的数据结构
    if topic_list is not None:
        # 使用分类的数据结构
        result = {}
        for data_type in topic_list.keys():
            result[data_type] = {}
    else:
        # 使用传统的数据结构
        images = {}  # {topic_name: [(timestamp, image_data), ...]}
        topics = {}  # {topic_name: [(timestamp, message_data), ...]}

    # 确定实际使用的时间范围（用于进度条显示）
    display_start_time = start_time
    display_end_time = end_time

    # print(f"len(time_ranges): {len(time_ranges)}")
    if time_ranges is not None and len(time_ranges) > 0:
        # 如果有多个时间范围，计算总体范围用于显示
        display_start_time = min(r[0] for r in time_ranges)
        display_end_time = max(r[1] for r in time_ranges)
    
    with open(mcap_path, "rb") as f:
        # 创建reader并注册decoder factory
        reader = make_reader(f, decoder_factories=[Ros2DecoderFactory()])
        
        # 使用 iter_decoded_messages() 直接获取解码后的消息
        # 返回 (schema, channel, message, ros_msg) 元组
        messages = reader.iter_decoded_messages()
        
        # 如果启用进度条，使用tqdm包装
        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(messages, desc="Loading mcap[", unit="] data, time used:", dynamic_ncols=True)
        else:
            pbar = messages
        
        message_count = 0
        skipped_before_start = 0
        skipped_after_end = 0
        consecutive_after_end = 0  # 连续超过end_time的消息数，用于提前退出
        last_progress_update_time = 0  # 上次更新进度条的时间（用于减少更新频率）
        
        for schema, channel, message, ros_msg in pbar:
            message_count += 1
            
            # 使用message的log_time作为时间戳（纳秒）
            timestamp_ns = message.log_time if hasattr(message, 'log_time') else 0
            timestamp_sec = timestamp_ns / 1e9
            
            # 检查时间戳是否在允许的时间范围内
            in_time_range = False
            
            if time_ranges is not None:
                # 使用多个时间范围（已按开始时间排序，且不重叠）
                for range_start, range_end in time_ranges:
                    # 如果时间戳小于当前范围的开始时间，可以提前退出
                    # 因为后续范围的开始时间会更大（已排序且不重叠）
                    if timestamp_sec < range_start:
                        break
                    # 如果时间戳在当前范围内，找到匹配
                    if timestamp_sec <= range_end:
                        in_time_range = True
                        break
                    # 如果时间戳大于当前范围的结束时间，继续检查下一个范围
            else:
                # 使用单个时间范围（start_time, end_time）
                in_time_range = True
                if start_time is not None and timestamp_sec < start_time:
                    in_time_range = False
                if end_time is not None and timestamp_sec > end_time:
                    in_time_range = False
            
            # 如果不在时间范围内，跳过
            if not in_time_range:
                if time_ranges is not None:
                    # 检查是在所有范围之前还是之后
                    if len(time_ranges) > 0 and timestamp_sec < time_ranges[0][0]:
                        skipped_before_start += 1
                    elif len(time_ranges) > 0 and timestamp_sec > time_ranges[-1][1]:
                        skipped_after_end += 1
                        consecutive_after_end += 1
                    else:
                        # 在时间范围之间的间隙
                        skipped_before_start += 1
                else:
                    # 使用单个时间范围
                    if start_time is not None and timestamp_sec < start_time:
                        skipped_before_start += 1
                    elif end_time is not None and timestamp_sec > end_time:
                        skipped_after_end += 1
                        consecutive_after_end += 1
                
                # 更新可视化进度条（每0.1秒更新一次，减少开销）
                if show_progress and TQDM_AVAILABLE:
                    if timestamp_sec - last_progress_update_time > 0.1:
                        time_info = create_time_progress_info(timestamp_sec, display_start_time, display_end_time, time_ranges)
                        pbar.set_postfix_str(time_info)
                        last_progress_update_time = timestamp_sec
                continue
            
            # 重置连续超过end_time的计数器（因为这条消息在时间范围内）
            consecutive_after_end = 0
            
            topic_name = channel.topic
            
            # 如果提供了topic_list，检查该topic是否在列表中
            if topic_list is not None:
                if topic_name not in topic_to_type:
                    # 不在列表中的topic，跳过
                    continue
                
                data_type = topic_to_type[topic_name]
                
                # 根据数据类型直接提取
                if data_type == "images":
                    img = decode_image_message(ros_msg)
                    if img is not None:
                        if topic_name not in result[data_type]:
                            result[data_type][topic_name] = []
                        result[data_type][topic_name].append((timestamp_sec, img))
                
                elif data_type == "pose7d":
                    pose7d = extract_pose_from_msg(ros_msg)
                    if pose7d is not None:
                        if topic_name not in result[data_type]:
                            result[data_type][topic_name] = []
                        result[data_type][topic_name].append((timestamp_sec, pose7d))
                
                elif data_type == "joint_states":
                    joint_states = extract_joint_state_from_msg(ros_msg)
                    if joint_states is not None:
                        if topic_name not in result[data_type]:
                            result[data_type][topic_name] = []
                        result[data_type][topic_name].append((timestamp_sec, joint_states))
                        
                elif data_type == "gripper":
                    grippers = extract_gripper_from_msg(ros_msg)
                    if grippers is not None:
                        if topic_name not in result[data_type]:
                            result[data_type][topic_name] = []
                        result[data_type][topic_name].append((timestamp_sec, grippers))
                    pass
                else:
                    raise ValueError(f"Unknown data type: {data_type}")

                # 更新进度条信息
                # if show_progress and TQDM_AVAILABLE:
                #     counts = {k: len(v) for k, v in result.items()}
                #     # 显示时间信息和统计数据
                #     if timestamp_sec - last_progress_update_time > 0.1:
                #         time_info = create_time_progress_info(timestamp_sec, display_start_time, display_end_time, time_ranges)
                #         postfix = {**counts, '当前': topic_name[:30], '时间': time_info}
                #         pbar.set_postfix(postfix)
                #         last_progress_update_time = timestamp_sec
                #     else:
                #         pbar.set_postfix({
                #             **counts,
                #             '当前': topic_name[:30]
                #         })
            else:
                # 传统模式：检查是否是图像消息（通过schema名称或topic名称）
                is_image = False
                if schema is not None:
                    schema_name = schema.name if hasattr(schema, 'name') else str(schema)
                    if 'Image' in schema_name or 'image' in topic_name.lower():
                        is_image = True
                
                # 处理图像消息
                if is_image:
                    img = decode_image_message(ros_msg)
                    if img is not None:
                        if topic_name not in images:
                            images[topic_name] = []
                        images[topic_name].append((timestamp_sec, img))
                        # # 更新进度条信息
                        # if show_progress and TQDM_AVAILABLE:
                        #     if timestamp_sec - last_progress_update_time > 0.1:
                        #         time_info = create_time_progress_info(timestamp_sec, display_start_time, display_end_time, time_ranges)
                        #         pbar.set_postfix({
                        #             '图像topics': len(images),
                        #             '其他topics': len(topics),
                        #             '当前': topic_name[:30],
                        #             '时间': time_info
                        #         })
                        #         last_progress_update_time = timestamp_sec
                        #     else:
                        #         pbar.set_postfix({
                        #             '图像topics': len(images),
                        #             '其他topics': len(topics),
                        #             '当前': topic_name[:30]
                        #         })
                else:
                    # 处理其他topic消息（如状态、动作等）
                    if topic_name not in topics:
                        topics[topic_name] = []
                    msg_data = extract_message_data(ros_msg)
                    topics[topic_name].append((timestamp_sec, msg_data))
                    # 更新进度条信息
                    # if show_progress and TQDM_AVAILABLE:
                    #     postfix = {
                    #         '图像topics': len(images),
                    #         '其他topics': len(topics),
                    #         '当前': topic_name[:30]
                    #     }
                    #     # 添加时间信息
                    #     if timestamp_sec - last_progress_update_time > 0.1:
                    #         time_info = create_time_progress_info(timestamp_sec, display_start_time, display_end_time, time_ranges)
                    #         postfix['时间'] = time_info
                    #         last_progress_update_time = timestamp_sec
                    #     pbar.set_postfix(postfix)
        
        # 检查是否读取到任何消息
        if message_count == 0:
            print("警告: 没有读取到任何消息。可能的原因:")
            print("  1. 文件为空或损坏")
            print("  2. decoder factory 未正确注册")
            print("  3. 文件格式不兼容")
            print(f"  文件路径: {mcap_path}")
        
        # 打印时间范围过滤统计信息
        if start_time is not None or end_time is not None:
            total_processed = message_count
            if start_time is not None:
                print(f"时间范围过滤: 跳过了 {skipped_before_start} 条在开始时间之前的消息")
            if end_time is not None:
                print(f"时间范围过滤: 跳过了 {skipped_after_end} 条在结束时间之后的消息")
            if total_processed > 0:
                kept_ratio = (total_processed - skipped_before_start - skipped_after_end) / total_processed * 100
                print(f"时间范围过滤: 保留了 {total_processed - skipped_before_start - skipped_after_end} / {total_processed} 条消息 ({kept_ratio:.1f}%)")
    
    # 返回相应的数据结构
    if topic_list is not None:
        return result
    else:
        
        return {
            'images': images,
            'topics': topics
        }


def extract_gripper_from_msg(msg) -> np.ndarray:
    if hasattr(msg, 'angle'):
        gripper_state = msg.angle
    else:
        raise ValueError("cannot extract gripper state from topic")
    return np.array([gripper_state], dtype=np.float32)


# 提取单条joint_states
def extract_joint_state_from_msg(msg) -> np.ndarray:
    if hasattr(msg, 'position'):
        assert len(msg.position) == 6, "joint_states position != 6"
        joint_state = msg.position
    else:
        raise ValueError("cannot extract joint_states from topic")
    return np.array(joint_state, dtype=np.float32)

# 提取单条pose6d
def extract_pose_from_msg(msg) -> np.ndarray:
    """
    从topic数据中提取pose数据。
    
    Args:
        topic_data: topic数据列表，格式为 [(timestamp, message_data), ...]
        topic_name: topic名称（用于错误提示）
        
    Returns:
        poses: numpy数组，形状为(N, 7)，包含[x, y, z, roll, pitch, yaw, gripper]
    """
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'position') and hasattr(msg.pose, 'orientation'):
        # PoseStamped格式
        roll, pitch, yaw = euler_from_quaternion([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        pose6d = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            roll,
            pitch,
            yaw
        ]
    else:
        raise ValueError("cannot extract pose from topic")
        
    return np.array(pose6d, dtype=np.float32)



def extract_pose_from_topic_data(topic_data: list, topic_name: str) -> np.ndarray:
    """
    从topic数据中提取pose数据。
    
    Args:
        topic_data: topic数据列表，格式为 [(timestamp, message_data), ...]
        topic_name: topic名称（用于错误提示）
        
    Returns:
        poses: numpy数组，形状为(N, 7)，包含[x, y, z, roll, pitch, yaw, gripper]
    """
    poses = []
    
    for timestamp, msg_data in topic_data:
        try:
            # 尝试提取PoseStamped格式的数据
            # geometry_msgs/PoseStamped 结构: pose.position, pose.orientation
            x, y, z = 0.0, 0.0, 0.0
            roll, pitch, yaw = 0.0, 0.0, 0.0
            gripper = 0.0
            
            # 方法1: 从嵌套的pose结构中提取
            if 'pose' in msg_data:
                pose = msg_data['pose']
                if isinstance(pose, dict):
                    # 提取位置
                    if 'position' in pose:
                        pos = pose['position']
                        if isinstance(pos, dict):
                            x = pos.get('x', 0.0)
                            y = pos.get('y', 0.0)
                            z = pos.get('z', 0.0)
                    # 提取姿态
                    if 'orientation' in pose:
                        orient = pose['orientation']
                        if isinstance(orient, dict):
                            qx = orient.get('x', 0.0)
                            qy = orient.get('y', 0.0)
                            qz = orient.get('z', 0.0)
                            qw = orient.get('w', 1.0)
                            roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
            # 方法2: 直接从顶层提取
            elif 'position' in msg_data and 'orientation' in msg_data:
                pos = msg_data['position']
                if isinstance(pos, dict):
                    x = pos.get('x', 0.0)
                    y = pos.get('y', 0.0)
                    z = pos.get('z', 0.0)
                orient = msg_data['orientation']
                if isinstance(orient, dict):
                    qx = orient.get('x', 0.0)
                    qy = orient.get('y', 0.0)
                    qz = orient.get('z', 0.0)
                    qw = orient.get('w', 1.0)
                    roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
            # 方法3: 直接字段访问
            elif 'x' in msg_data and 'y' in msg_data and 'z' in msg_data:
                x = msg_data.get('x', 0.0)
                y = msg_data.get('y', 0.0)
                z = msg_data.get('z', 0.0)
                if 'roll' in msg_data and 'pitch' in msg_data and 'yaw' in msg_data:
                    roll = msg_data.get('roll', 0.0)
                    pitch = msg_data.get('pitch', 0.0)
                    yaw = msg_data.get('yaw', 0.0)
                elif 'qx' in msg_data or 'orientation_x' in msg_data:
                    qx = msg_data.get('qx', msg_data.get('orientation_x', 0.0))
                    qy = msg_data.get('qy', msg_data.get('orientation_y', 0.0))
                    qz = msg_data.get('qz', msg_data.get('orientation_z', 0.0))
                    qw = msg_data.get('qw', msg_data.get('orientation_w', 1.0))
                    roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
            
            # 提取gripper状态（如果存在）
            if 'gripper' in msg_data:
                gripper = msg_data['gripper']
            elif 'gripper_state' in msg_data:
                gripper = msg_data['gripper_state']
            elif 'gripper_position' in msg_data:
                gripper = msg_data['gripper_position']
            
            poses.append([x, y, z, roll, pitch, yaw, gripper])
            
        except Exception as e:
            print(f"警告: 解析消息时出错 (timestamp={timestamp}): {e}")
            print(f"  消息数据: {list(msg_data.keys())[:10]}...")  # 只显示前10个键
            continue
    
    if len(poses) == 0:
        raise ValueError(f"无法从topic '{topic_name}'中提取pose数据")
    
    return np.array(poses, dtype=np.float32)


def list_mcap_topics_from_data(mcap_data: dict) -> dict:
    """
    从mcap数据中列出所有topic及其统计信息。
    
    Args:
        mcap_data: read_mcap_file返回的数据
        
    Returns:
        topic_info: 字典，key为topic名称，value为统计信息
    """
    topic_info = {}
    
    # 处理图像topics
    for topic_name, image_list in mcap_data.get('images', {}).items():
        if len(image_list) > 0:
            timestamps = [ts for ts, _ in image_list]
            time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
            fps = len(image_list) / time_span if time_span > 0 else 0.0
            
            topic_info[topic_name] = {
                "count": len(image_list),
                "fps": fps,
                "type": "image",
                "time_span": time_span
            }
    
    # 处理其他topics
    for topic_name, topic_list in mcap_data.get('topics', {}).items():
        if len(topic_list) > 0:
            timestamps = [ts for ts, _ in topic_list]
            time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
            fps = len(topic_list) / time_span if time_span > 0 else 0.0
            
            # 计算平均时间间隔
            if len(timestamps) > 1:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = np.mean(intervals) if intervals else 0.0
                avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
            else:
                avg_fps = 0.0
            
            topic_info[topic_name] = {
                "count": len(topic_list),
                "fps": fps,
                "avg_fps": avg_fps,
                "type": "data",
                "time_span": time_span
            }
    
    return topic_info

def quaternion_to_euler(qx, qy, qz, qw):
    """将四元数转换为欧拉角（roll, pitch, yaw），单位：弧度
    先转换为3x3旋转矩阵，再从旋转矩阵提取欧拉角"""
    # 将四元数转换为3x3旋转矩阵
    # 四元数 q = (qx, qy, qz, qw)
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qw2 = qw * qw
    
    # 旋转矩阵 R
    R = np.array([
        [1 - 2*(qy2 + qz2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx2 + qz2),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx2 + qy2)]
    ])
    
    # 从旋转矩阵提取欧拉角（ZYX顺序：yaw-pitch-roll）
    # R = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
    
    # 计算yaw（绕Z轴旋转）
    yaw = np.arctan2(r21, r11)
    
    # 计算pitch（绕Y轴旋转）
    sinp = -r31
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # 计算roll（绕X轴旋转）
    roll = np.arctan2(r32, r33)
    
    return np.array([roll, pitch, yaw])


def euler_from_quaternion(quaternion):
    """
    将四元数转换为欧拉角（roll, pitch, yaw），单位：弧度
    
    Args:
        quaternion: 四元数 [x, y, z, w] 或 (x, y, z, w)
        
    Returns:
        (roll, pitch, yaw): 欧拉角，单位：弧度
    """
    qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    euler = quaternion_to_euler(qx, qy, qz, qw)
    return euler[0], euler[1], euler[2]  # roll, pitch, yaw


def read_pose_from_mcap(mcap_path: str, topic: str = "/pika_pose_r") -> np.ndarray:
    """
    从mcap文件中读取指定topic的pose数据，并将四元数转换为欧拉角。
    
    Args:
        mcap_path: mcap文件路径
        topic: 要读取的topic名称，默认为"/pika_pose_r"
        
    Returns:
        trajectory: numpy数组，形状为(N, 6)，包含[x, y, z, roll, pitch, yaw]
    """
    if not MCAP_AVAILABLE:
        raise ImportError("mcap库未安装，请安装: pip install mcap mcap-ros2-support")
    
    trajectory = []
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[Ros2DecoderFactory()])
        
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            if channel.topic == topic:
                msg = ros_msg
                
                # 提取位置和姿态
                if hasattr(msg, 'pose') and hasattr(msg.pose, 'position') and hasattr(msg.pose, 'orientation'):
                    # PoseStamped格式
                    roll, pitch, yaw = euler_from_quaternion([
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w
                    ])
                    trajectory.append([
                        msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z,
                        roll,
                        pitch,
                        yaw
                    ])
                elif hasattr(msg, 'position') and hasattr(msg, 'orientation'):
                    # 直接Pose格式
                    roll, pitch, yaw = euler_from_quaternion([
                        msg.orientation.x,
                        msg.orientation.y,
                        msg.orientation.z,
                        msg.orientation.w
                    ])
                    trajectory.append([
                        msg.position.x,
                        msg.position.y,
                        msg.position.z,
                        roll,
                        pitch,
                        yaw
                    ])
                else:
                    print(f"警告: 无法提取pose信息，跳过此消息")
                    continue
    
    if len(trajectory) == 0:
        raise ValueError(f"未找到topic '{topic}'的数据")
    
    trajectory = np.array(trajectory)
    print(f"成功读取 {len(trajectory)} 个pose数据")
    return trajectory


def extract_joint_states_from_topic_data(topic_data: list, topic_name: str) -> list:
    """
    从topic数据中提取joint_states数据（6维position + 6维velocity）。
    
    Args:
        topic_data: topic数据列表，格式为 [(timestamp, message_data), ...]
        topic_name: topic名称（用于错误提示）
        
    Returns:
        joint_states: 列表，格式为 [(timestamp, np.array([pos_6d, vel_6d])), ...]
        其中 pos_6d 是6维位置，vel_6d 是6维速度，总共12维
    """
    joint_states = []
    
    for timestamp, msg_data in topic_data:
        try:
            position = np.zeros(6, dtype=np.float32)
            velocity = np.zeros(6, dtype=np.float32)
            
            # 方法1: 从标准的 JointState 消息结构提取
            # sensor_msgs/JointState 结构: position, velocity, name
            if 'position' in msg_data:
                pos_data = msg_data['position']
                if isinstance(pos_data, (list, np.ndarray)):
                    pos_array = np.array(pos_data, dtype=np.float32)
                    # 取前6维
                    position = pos_array[:6] if len(pos_array) >= 6 else np.pad(pos_array, (0, 6 - len(pos_array)), 'constant')[:6]
            
            if 'velocity' in msg_data:
                vel_data = msg_data['velocity']
                if isinstance(vel_data, (list, np.ndarray)):
                    vel_array = np.array(vel_data, dtype=np.float32)
                    # 取前6维
                    velocity = vel_array[:6] if len(vel_array) >= 6 else np.pad(vel_array, (0, 6 - len(vel_array)), 'constant')[:6]
            
            # 方法2: 尝试从其他可能的字段名提取
            if len(position) == 0 or np.all(position == 0):
                # 尝试其他可能的字段名
                for key in ['positions', 'joint_positions', 'pos', 'joint_pos']:
                    if key in msg_data:
                        pos_data = msg_data[key]
                        if isinstance(pos_data, (list, np.ndarray)):
                            pos_array = np.array(pos_data, dtype=np.float32)
                            position = pos_array[:6] if len(pos_array) >= 6 else np.pad(pos_array, (0, 6 - len(pos_array)), 'constant')[:6]
                            break
            
            if len(velocity) == 0 or np.all(velocity == 0):
                # 尝试其他可能的字段名
                for key in ['velocities', 'joint_velocities', 'vel', 'joint_vel']:
                    if key in msg_data:
                        vel_data = msg_data[key]
                        if isinstance(vel_data, (list, np.ndarray)):
                            vel_array = np.array(vel_data, dtype=np.float32)
                            velocity = vel_array[:6] if len(vel_array) >= 6 else np.pad(vel_array, (0, 6 - len(vel_array)), 'constant')[:6]
                            break
            
            # 合并为12维向量 [position_6d, velocity_6d]
            joint_state = np.concatenate([position, velocity], axis=0)
            joint_states.append((timestamp, joint_state))
            
        except Exception as e:
            print(f"警告: 解析joint_states消息时出错 (timestamp={timestamp}): {e}")
            print(f"  消息数据键: {list(msg_data.keys())[:10]}...")  # 只显示前10个键
            continue
    
    if len(joint_states) == 0:
        print(f"警告: 无法从topic '{topic_name}'中提取joint_states数据")
    
    return joint_states

def load_mcap_data(mcap_path: str, arm_topics,
                   start_time: Optional[float] = None, end_time: Optional[float] = None,
                   segments: Optional[List[Dict]] = None, buffer_seconds: float = 3.0) -> Dict[str, Dict[str, list]]:
    """
    从mcap文件中加载指定topic的数据，支持时间范围过滤
    
    Args:
        mcap_path: mcap文件路径
        arm_topics: ArmTopics实例，包含所有关节的topic名称
        start_time: 可选，开始时间（秒），只加载此时间之后的数据（当segments为None时使用）
        end_time: 可选，结束时间（秒），只加载此时间之前的数据（当segments为None时使用）
        segments: 可选，从JSON读取的segments列表，每个segment包含startSec和endSec字段。
                  如果提供，将仅加载segments时间范围内的数据（前后冗余buffer_seconds秒）
        buffer_seconds: 当使用segments时，在每个segment前后冗余加载的秒数，默认3.0秒
        
    Returns:
        加载后的数据字典
    """
    if not MCAP_AVAILABLE:
        raise ImportError("mcap库未安装 请安装: pip install mcap mcap-ros2-support")
    
    topic_list = arm_topics.to_topic_list()
    # 如果提供了segments，计算需要加载的时间范围
    if segments is not None:
        time_ranges = []
        
        # 收集所有segment的时间范围（包括subtasks）
        for segment in segments:
            start_sec = segment.get('startSec')
            end_sec = segment.get('endSec')
            
            if start_sec is None or end_sec is None:
                continue
            if end_sec <= start_sec:
                continue
            
            # 检查是否有subtasks
            subtasks = segment.get('subtasks', [])
            if len(subtasks) == 0:
                # 没有subtasks，使用segment本身
                # 添加buffer：前后各冗余buffer_seconds秒
                time_ranges.append((max(0, start_sec - buffer_seconds), end_sec + buffer_seconds))
            else:
                # 有subtasks，收集每个subtask的时间范围
                for subtask_meta in subtasks:
                    subtask_start_sec = subtask_meta.get('startSec', None)
                    subtask_end_sec = subtask_meta.get('endSec', None)
                    if subtask_start_sec is None or subtask_end_sec is None:
                        continue
                    if subtask_end_sec <= subtask_start_sec:
                        continue
                    
                    # 添加buffer：前后各冗余buffer_seconds秒
                    time_ranges.append((max(0, subtask_start_sec - buffer_seconds), subtask_end_sec + buffer_seconds))
        
        if not time_ranges:
            raise ValueError("错误: segments中没有有效的时间范围")
        
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
        
        # 确保merged_ranges按开始时间排序（虽然合并算法已经保证有序，但显式排序更安全）
        merged_ranges.sort(key=lambda x: x[0])
        
        # 计算总时间范围（仅用于显示）
        overall_start = min(r[0] for r in merged_ranges)
        overall_end = max(r[1] for r in merged_ranges)
        total_duration = sum(r[1] - r[0] for r in merged_ranges)
        
        print(f"从segments计算得到 {len(merged_ranges)} 个时间片段，总时长: {total_duration:.2f}s")
        print(f"ts range: {overall_start:.2f}s - {overall_end:.2f}s (range: {overall_end - overall_start:.2f}s)")
        print(f"每个segment前后冗余 {buffer_seconds} 秒")
        
        # 将merged_ranges传递给read_mcap_file，只加载这些片段内的数据
        result = read_mcap_file(mcap_path, show_progress=True, topic_list=topic_list, 
                               time_ranges=merged_ranges)
    else:
        # 如果没有提供segments，使用start_time和end_time
        if start_time is not None and end_time is not None:
            if end_time <= start_time:
                raise ValueError(f"错误: end_time ({end_time}) 必须大于 start_time ({start_time})")
            
            # 添加buffer：前后各冗余buffer_seconds秒
            actual_start_time = max(0, start_time - buffer_seconds)
            actual_end_time = end_time + buffer_seconds
            
            print(f"使用时间范围: {start_time:.2f}s - {end_time:.2f}s")
            print(f"实际加载范围: {actual_start_time:.2f}s - {actual_end_time:.2f}s (前后冗余 {buffer_seconds} 秒)")
            
            result = read_mcap_file(mcap_path, show_progress=True, topic_list=topic_list,
                                   start_time=actual_start_time, end_time=actual_end_time)
        else:
            raise ValueError("必须提供segments参数，或者同时提供start_time和end_time参数以加载mcap数据")
    
    # 确保所有topic都有对应的数据（即使为空列表）
    for data_type, topics in topic_list.items():
        if data_type not in result:
            result[data_type] = {}
        for topic_name in topics:
            if topic_name not in result[data_type]:
                print(f"WARNNING: cannot found {data_type} topic '{topic_name}'")
                result[data_type][topic_name] = []
    
    # 打印所有topic的数据量
    # 并使用result['images']['/gripper/camera_fisheye_r/color/image_raw'][0][0]和result['images']['/gripper/camera_fisheye_r/color/image_raw'][-1][0]计算时间范围，计算帧率
    for data_type, topics_data in result.items():
        for topic_name, data_list in topics_data.items():
            if len(data_list) > 1:
                actual_start_time = data_list[0][0]
                actual_end_time = data_list[-1][0]
                time_span = actual_end_time - actual_start_time
                fps = len(data_list) / time_span if time_span > 0 else 0.0
                print(f"Load {data_type} topic '{topic_name}', data size: {len(data_list)}, ts range: {time_span:.2f}s, FPS: {fps:.2f} Hz")
            else:
                print(f"Load {data_type} topic '{topic_name}', data size: {len(data_list)}")
    return result
