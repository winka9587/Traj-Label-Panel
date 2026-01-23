
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


from typing import Dict, List, Optional, Any
def read_mcap_file(mcap_path: str, show_progress: bool = True, topic_list: Optional[Dict[str, List[str]]] = None, 
                   start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict:
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
        start_time: 可选，开始时间（秒），只加载此时间之后的数据
        end_time: 可选，结束时间（秒），只加载此时间之前的数据
        
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

    with open(mcap_path, "rb") as f:
        # 创建reader并注册decoder factory
        reader = make_reader(f, decoder_factories=[Ros2DecoderFactory()])
        
        # 使用 iter_decoded_messages() 直接获取解码后的消息
        # 返回 (schema, channel, message, ros_msg) 元组
        messages = reader.iter_decoded_messages()
        
        # 如果启用进度条，使用tqdm包装
        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(messages, desc="读取mcap文件", unit="消息", dynamic_ncols=True)
        else:
            pbar = messages
        
        message_count = 0
        for schema, channel, message, ros_msg in pbar:
            message_count += 1
            
            # 使用message的log_time作为时间戳（纳秒）
            timestamp_ns = message.log_time if hasattr(message, 'log_time') else 0
            timestamp_sec = timestamp_ns / 1e9
            
            # 如果指定了时间范围，过滤数据
            if start_time is not None and timestamp_sec < start_time:
                continue
            if end_time is not None and timestamp_sec > end_time:
                continue
            
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
                if show_progress and TQDM_AVAILABLE:
                    counts = {k: len(v) for k, v in result.items()}
                    pbar.set_postfix({
                        **counts,
                        '当前': topic_name[:30]
                    })
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
                        # 更新进度条信息
                        if show_progress and TQDM_AVAILABLE:
                            pbar.set_postfix({
                                '图像topics': len(images),
                                '其他topics': len(topics),
                                '当前': topic_name[:30]
                            })
                else:
                    # 处理其他topic消息（如状态、动作等）
                    if topic_name not in topics:
                        topics[topic_name] = []
                    msg_data = extract_message_data(ros_msg)
                    topics[topic_name].append((timestamp_sec, msg_data))
                    # 更新进度条信息
                    if show_progress and TQDM_AVAILABLE:
                        pbar.set_postfix({
                            '图像topics': len(images),
                            '其他topics': len(topics),
                            '当前': topic_name[:30]
                        })
        
        # 检查是否读取到任何消息
        if message_count == 0:
            print("警告: 没有读取到任何消息。可能的原因:")
            print("  1. 文件为空或损坏")
            print("  2. decoder factory 未正确注册")
            print("  3. 文件格式不兼容")
            print(f"  文件路径: {mcap_path}")
    
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
    """将四元数转换为欧拉角（roll, pitch, yaw），单位：弧度"""
    # 绕X轴旋转（roll）
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # 绕Y轴旋转（pitch）
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # 绕Z轴旋转（yaw）
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
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


def load_mcap_data(mcap_path: str, topic_list: Dict[str, List[str]], 
                   start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict[str, Dict[str, list]]:
    """
    从mcap文件中加载指定topic的数据，支持时间范围过滤
    
    Args:
        mcap_path: mcap文件路径
        topic_list: topic列表字典，key是数据类型，value是topic名称列表
        start_time: 可选，开始时间（秒），只加载此时间之后的数据
        end_time: 可选，结束时间（秒），只加载此时间之前的数据
        
    Returns:
        加载后的数据字典
    """
    if not MCAP_AVAILABLE:
        raise ImportError("mcap库未安装 请安装: pip install mcap mcap-ros2-support")
    
    # 直接使用read_mcap_file读取并提取数据
    result = read_mcap_file(mcap_path, show_progress=True, topic_list=topic_list, 
                           start_time=start_time, end_time=end_time)
    
    # 确保所有topic都有对应的数据（即使为空列表）
    for data_type, topics in topic_list.items():
        if data_type not in result:
            result[data_type] = {}
        for topic_name in topics:
            if topic_name not in result[data_type]:
                print(f"警告: 未找到{data_type} topic '{topic_name}'")
                result[data_type][topic_name] = []
    
    # 打印所有topic的数据量
    # 并使用result['images']['/gripper/camera_fisheye_r/color/image_raw'][0][0]和result['images']['/gripper/camera_fisheye_r/color/image_raw'][-1][0]计算时间范围，计算帧率
    for data_type, topics_data in result.items():
        for topic_name, data_list in topics_data.items():
            if len(data_list) > 1:
                start_time = data_list[0][0]
                end_time = data_list[-1][0]
                time_span = end_time - start_time
                fps = len(data_list) / time_span if time_span > 0 else 0.0
                print(f"已加载 {data_type} topic '{topic_name}'，数据量: {len(data_list)}，时间范围: {time_span:.2f}s，帧率: {fps:.2f} Hz")
            else:
                print(f"已加载 {data_type} topic '{topic_name}'，数据量: {len(data_list)}")
    return result
