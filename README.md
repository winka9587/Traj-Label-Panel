# umi_cut

![](/assets/img/2026-01-26-15-01-38.png)

Lichtblick extension，提供针对mcap格式遥操数据的标注插件

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
6. :construction: 加载已有json文件，检查/修改标注结果
7. :black_square_button: 撤回上一个标注动作

## env

~~~

# script/gen_lerobot_dataset 依赖
pip install mcap mcap-ros2-support
~~~

## 注意

目前的script中的脚本没有添加对 segments.json中重叠的处理（假设：一个json文件中的所有分段是不会出现重叠的）