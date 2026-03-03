# umi_cut

参考: https://github.com/Lichtblick-Suite/create-lichtblick-extension

![](/assets/img/2026-01-26-15-01-38.png)

Lichtblick extension，提供针对mcap格式录制数据的标注插件

| 图标 | 说明 |
|----------|----------|
| :white_check_mark: | **基本完成** 不排除有新增任务的可能 |
| :black_square_button: | **待做** |
| :heavy_check_mark: | 可预见的未来 **不需要进行任何修改** |
| :construction: | 近期（2周内）**正在处理** |
| :warning: | **实现有待讨论** |

features:
1. :white_check_mark: 读取mcap数据进行数据标注，生成标注json文件
2. :white_check_mark: 添加segment可视化
3. :white_check_mark: 提供python数据处理脚本，加载mcap文件和json文件，生成处理后的lerobot数据
4. :white_check_mark: 支持task+subtask预设
5. :white_check_mark: 交互修改已生成的标注
6. :white_check_mark: 加载已有json文件，检查/修改标注结果
7. :black_square_button: 撤回上一个标注动作

## env

~~~

# script/gen_lerobot_dataset 依赖
pip install mcap mcap-ros2-support
~~~

## install

~~~
npm run build
npm run package　# 生成.foxe 
~~~

./dist下会生成extension.js

foxe文件本质是一个压缩包，将其内容解压到~/.lichtblick-suite/extensions即可

~~~
~/.lichtblick-suite
└── extensions
    └── unknown.umi_cut-0.1.0
        ├── CHANGELOG.md
        ├── dist
        │   └── extension.js
        ├── package.json
        └── README.md
~~~

添加了对lerobot的修改来跳过split生成，加速数据集的本地构建。如果需要该功能，拉取并使用修改后的lerobot：
~~~
git clone git@github.com:winka9587/lerobot-for-DataGen.git
cd lerobot-for-DataGen
pip install -e .
~~~


## 注意

目前的script中的脚本没有添加对 segments.json中重叠的处理（假设：一个json文件中的所有分段是不会出现重叠的）

### 关于 `lerobot-edit-dataset` 与删除 episode

文档中的删除示例：

```bash
lerobot-edit-dataset --repo_id lerobot/pusht --operation.type delete_episodes --operation.episode_indices "[0, 2, 5]"
```

**该命令仅在 lerobot 0.4+ 中提供。** 若你当前是 `lerobot 0.1.0`（`pip list | grep lerobot`），则不会包含 `lerobot-edit-dataset`，因此会报「未找到命令」。

可选做法：

1. **升级 lerobot 后使用官方命令**（推荐，若可接受升级）：
   ```bash
   pip install -U lerobot
   lerobot-edit-dataset --repo_id <你的repo_id> --operation.type delete_episodes --operation.episode_indices "[0, 2, 5]"
   ```
   若数据集在本地，可加 `--root /path/to/HF_LEROBOT_HOME`。

2. **不升级时**：使用本仓库提供的脚本（会检测是否有 `lerobot-edit-dataset`，没有则提示升级）：
   ```bash
   cd script && python delete_episodes_lerobot.py --repo_id pick_cillion_umi_v0 --episode_indices "[0, 2, 5]"
   ```
   **注意**：在 0.1.0 下该脚本只会提示升级，不会执行删除；升级到 0.4+ 后同一命令会调用官方工具执行删除。

## 

~~~
┌─────────────────────────────────────┐
│ 程序启动 (main函数) │
│ 解析命令行参数 │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ create_lerobot_dataset() │
│ 1. 检查文件是否存在 │
│ - mcap_path │
│ - segments_json_path │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 加载segments.json │
│ load_segments_json() │
│ 获取所有segment信息 │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 解析segments │
│ 根据annotation_level处理: │
│ ┌─────────────────────────────┐ │
│ │ full_task: 整个任务 │ │
│ │ sub_task: 子任务 │ │
│ └─────────────────────────────┘ │
│ 生成segment_info_list │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 初始化LeRobotDataset │
│ ┌─────────────────────────────┐ │
│ │ 如果存在: 加载数据集 │ │
│ │ 如果不存在: 创建新数据集 │ │
│ └─────────────────────────────┘ │
│ 配置features: │
│ - wrist_image_left/right │
│ - state (7维) │
│ - actions (7维) │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 遍历每个segment │
│ for seg_info in segment_info_list │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 检查是否已处理 │
│ check_already_processed() │
│ ┌─────────────────────────────┐ │
│ │ 1. 计算文件哈希 │ │
│ │ - mcap_hash │ │
│ │ - segments_hash │ │
│ │ 2. 组合哈希标识 │ │
│ │ 3. 查询处理记录 │ │
│ └─────────────────────────────┘ │
└──────────────┬──────────────────────┘
│
┌──────────────┴──────────────┐
│ │
▼ ▼
┌──────────────────┐ ┌──────────────────────┐
│ 已处理且存在 │ │ 未处理或需要重新处理│
│ 跳过此segment │ │ 继续处理 │
└──────────────────┘ └──────────┬───────────┘
│
▼
┌─────────────────────────────────────┐
│ 从mcap加载数据 │
│ load_mcap_data() │
│ ┌─────────────────────────────┐ │
│ │ 加载时间范围: │ │
│ │ start_sec ~ end_sec │ │
│ │ 前后冗余2秒用于插值 │ │
│ └─────────────────────────────┘ │
│ 提取topics: │
│ - pose7d (左右手位姿) │
│ - joint_states (关节状态) │
│ - images (左右图像) │
│ - gripper (夹爪数据) │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 数据同步和插值 │
│ sync_topic_data() │
│ ┌─────────────────────────────┐ │
│ │ 1. 对齐时间戳 │ │
│ │ 2. 插值到目标fps │ │
│ │ 3. 生成同步数据列表 │ │
│ └─────────────────────────────┘ │
│ 输出: │
│ - timestamp_list │
│ - image_list │
│ - state_list │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 处理每一帧数据 │
│ for i in range(len(image_list)) │
│ ┌─────────────────────────────┐ │
│ │ 1. 提取图像 │ │
│ │ - wrist_image_left │ │
│ │ - wrist_image_right │ │
│ │ 2. 调整图像尺寸 (640x480) │ │
│ │ 3. 旋转图像 (90度顺时针) │ │
│ │ 4. 提取state (7维) │ │
│ │ 5. 生成action │ │
│ │ (current_state模式) │ │
│ │ 6. 构建task字符串 │ │
│ │ '{taskId}-{prompt}' │ │
│ └─────────────────────────────┘ │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 添加到数据集 │
│ dataset.add_frame() │
│ 包含: │
│ - wrist_image_left │
│ - wrist_image_right │
│ - state │
│ - actions │
│ - task │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 保存episode │
│ dataset.save_episode() │
│ 释放内存: │
│ - del image_list │
│ - del state_list │
│ - del timestamp_list │
│ - gc.collect() │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 更新处理记录 │
│ update_processed_files() │
│ ┌─────────────────────────────┐ │
│ │ 1. 计算文件哈希 │ │
│ │ 2. 创建处理记录 │ │
│ │ 3. 保存到processed_files.json│ │
│ │ (原子性写入) │ │
│ └─────────────────────────────┘ │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 继续下一个segment │
│ (循环处理) │
└──────────────┬──────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 显示统计信息 │
│ - 总帧数 │
│ - Episode数量 │
│ - 处理成功/跳过数量 │
└─────────────────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ 完成！返回结果 │
└─────────────────────────────────────┘
~~~