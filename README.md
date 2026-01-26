# umi_cut

Lichtblick extension，提供一个针对遥操数据的标注工具

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
4. 加载已有json文件，检查/修改标注结果

## env

~~~

# script/gen_lerobot_dataset 依赖
pip install mcap mcap-ros2-support
~~~

![](/assets/img/2026-01-26-15-01-38.png)