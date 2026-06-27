# lblStatistic

基于标签的指标统计

## Usage

### 评估每个类别的最优置信度阈值

```python
from lblStatistic import StatisticMatrix

# 获取检测的最优置信度阈值
stm = StatisticMatrix(
    pred_dir=["datasets/coco128/jsons"],  # list[str|Path]
    gt_dir=["datasets/coco128/xml1"],     # list[str|Path]
    dst_dir=Path("test_lbl_statistic"),   # 输出的保存根目录地址
    project="detection",                  # 实验项目名称, 用于生成保存文件夹
    plot=True,                            # 是否生成可视化结果
    classes=classes,                      # 类别名列表或者字典(编码与name的映射关系)
    pred_suffix=".json",                  # 预测结果文件后缀, Literal['.txt', '.json', '.xml']
    gt_suffix=".xml",                     # 可选类型分别表示yolo、labelme、pascalvoc的格式
    verbose=False,                        # 是否输出每个类别的详细信息
    chinese=False                         # 可以直接指定中文路径
)
stm()
```

**超参数**:

- `pred_dir`: 预测结果文件夹路径, 可以是多个文件夹路径的列表
- `gt_dir`: 真实标注文件夹路径, 可以是多个文件夹路径的列表
- `dst_dir`: 输出的保存根目录地址
- `project`: 实验项目名称, 用于生成保存文件夹
- `plot`: 是否生成可视化结果
- `classes`: 类别名列表或者字典(编码与name的映射关系)
- `pred_suffix`: 预测结果文件后缀, Literal['.txt', '.json', '.xml']分别表示yolo、labelme、pascalvoc的格式
- `gt_suffix`: 同上, 可配置不同的后缀
- `verbose`: 是否输出每个类别的详细信息
- `chinese`: 可以直接指定中文路径

**运行时参数**: 无

---

### 使用自定义IOS/IOU匹配规则统计指标

```python
from lblStatistic import StatisticConfusion

stc = StatisticConfusion(
    src_gt=["/Users/elfindan/datasets/coco128/jsons"],
    src_pred=["/Users/elfindan/datasets/coco128/xml1"],
    dst_dir=Path("test_lbl_statistic"),
    project='inference',
    use_ios=True,
    classes=classes,
    chinese=True,
    gt_suffix='.json',
    pred_suffix='.xml',
    use_fpfn=False,
    conf=0.,
    ios_thresh=0.1,
    iou_thresh=0.5,
    filter_category=[],
    difficult_filter=True,
)
stc()
```

**超参数**:

- src_gt (list[str]): 标签文件路径, 支持多个数据子集，需要指定到数据子集的标签文件存放路径
- src_pred (list[str]): 推理结果文件路径, 支持多个数据子集，需要指定到数据子集的推理结果文件存放路径
- dst_dir (str): 预测结果保存目录
- project (str, optional): 实验名称, defaults to 'inference'.
- use_ios (bool, optional): 是否使用IOS计算, defaults to True, False表示使用IOU计算
- classes (str, List[str]): 类别文件路径, defaults to 'classes.txt'.
- chinese (bool, optional): 是否使用中文类别, 可以指定中文字体文件的路径, defaults to True
- gt_suffix (str, optional): 标签文件后缀名, defaults to '.txt'.
- pred_suffix (str, optional): 推理结果文件后缀名, defaults to '.json'.
- use_fpfn (bool, optional): 是否保存FPFN的图片, defaults to False.
- conf (float, List[float]): 置信度阈值, defaults to 0.0.
- ios_thresh (float, optional): IOS阈值, defaults to 0.5.
- iou_thresh (float, optional): IOU阈值, defaults to 0.5.
- filter_category (List[str], optional): 过滤类别, defaults to [].
- difficult_filter (bool, optional): 是否过滤困难样本, defaults to True.

**运行时参数**:

- max_workers (int, optional): 并行处理的线程数, 默认是根据系统自动配置.
