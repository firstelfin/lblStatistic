#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2026/05/08 20:32:30

import os
import sys
import warnings
import threading
import numpy as np
from pprint import pprint
from loguru import logger
from pathlib import Path
from copy import deepcopy
from typing import Literal, Union, Dict, List, Optional
from prettytable import PrettyTable
from lblConvert.tools import read_yolo, read_voc, parser_json, read_txt, save_labelme_label
from lblConvert.tools.lblTools import FutureBar
from lblStatistic.utils.varType import PathStr
from lblStatistic.utils import set_plt, box_valid, path_list_valid, get_exp_dir
from lblStatistic.utils import DetMetrics, iou_np_boxes, ios_box, iou_box, xywh2xyxy
from .matrixBase import ConfusionMatrix

warnings.filterwarnings('ignore')


def identity(x):
    return x


def obj_matcher(
        pred_boxes: list[dict], gt_boxes: list[dict], iou_thresh=0.5, ios_thresh=0.5, 
        use_ios=False, mode="xyxy", classes: Optional[list]=None
    ):
    """yolo格式的预测框和真值框的匹配, 返回匹配结果

    ### Keyword
        - pred_boxes、gt_boxes: 若每个元素为 [类别, x, y, w, h] , 必须使用mode="xywh"

    :param pred_boxes: 预测框列表
    :type pred_boxes: list[dict]
    :param gt_boxes: gt框列表
    :type gt_boxes: list[dict]
    :param iou_thresh: IOU阈值, defaults to 0.5
    :type iou_thresh: float, optional
    :param ios_thresh: IOS阈值, defaults to 0.5
    :type ios_thresh: float, optional
    :param use_ios: 是否使用IOS匹配, defaults to False
    :type use_ios: bool, optional
    :param mode: 预测框和真值框的格式, "xyxy"表示(x1,y1,x2,y2), defaults to "xyxy"
    :type mode: str, optional
    :return: tpg, tpp, fp, fn
    :rtype: dict
    """

    # 格式转换
    trans_func = identity if mode.lower() == "xyxy" else xywh2xyxy
    box1_list = [trans_func(ins_obj["bbox"]) for ins_obj in pred_boxes]
    box2_list = [trans_func(ins_obj["bbox"]) for ins_obj in gt_boxes]
    pred_labels = np.array([ins_obj["label"] for ins_obj in pred_boxes])
    gt_labels = np.array([ins_obj["label"] for ins_obj in gt_boxes])
    # 基于GT记录difficult信息
    gt_difficult = np.array([ins_obj["flags"].get("difficult") for ins_obj in gt_boxes], dtype=bool)

    # 选择使用的匹配函数和阈值
    if use_ios:
        iou_func = ios_box
        thresh = ios_thresh
    else:
        iou_func = iou_box
        thresh = iou_thresh
    
    # 遍历预测框和真值框，计算IOU，并更新匹配状态

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    cls_matrix = pred_labels[:, None] == gt_labels[None, :]
    
    # 需要兼容IOS、IOU的计算, 不能使用numpy的广播机制
    for i, p_box in enumerate(box1_list):
        for j, g_box in enumerate(box2_list):
            iou = iou_func(g_box, p_box)
            iou_matrix[i, j] = iou
    iou_status_matrix = iou_matrix > thresh

    # 计算pred2gt_matrix
    pred2gt_matrix = iou_status_matrix * cls_matrix

    gt_status = np.any(pred2gt_matrix, axis=0)
    pred_status = np.any(pred2gt_matrix, axis=1)

    # 输出预测框和真值框的匹配情况
    match_object = {
        "boxesStatus": {
            "tpg": [gt_boxes[i] for i, status in enumerate(gt_status) if status],         # GT命中的框
            "tpp": [pred_boxes[i] for i, status in enumerate(pred_status) if status],     # pred命中的框
            "fp": [pred_boxes[i] for i, status in enumerate(pred_status) if not status],  # pred没有命中的框
            "fn": [gt_boxes[i] for i, status in enumerate(gt_status) if not (status or gt_difficult[i])],      # GT没有命中的框
            "fnDiff": [gt_boxes[i] for i, status in enumerate(gt_status) if not status and gt_difficult[i]],
            "tpDiff": [gt_boxes[i] for i, status in enumerate(gt_status) if status and gt_difficult[i]],
        },
        "updateItemsRecall": {},  # 记录召回, 按列更新
        "updateItemsPrecision": {},  # 记录精度, 按行更新
    }
    if classes is None:
        return match_object

    # 获取每一列最大值的索引和值
    _classes = deepcopy(classes)
    if _classes[-1] != 'background':
        _classes.append('background')
    update_items_recall = {class_name: [0]*len(_classes) for class_name in _classes}
    update_items_precision = {class_name: [0]*len(_classes) for class_name in _classes}
    # Note: 记录fn、tpg数据；fn也即将instance预测为background, tpg是gt中和预测完美匹配的实例
    target_items_after_filtered = np.array([True]*len(gt_boxes), dtype=bool)  # 需要纳入GT考虑的范围
    target_gt_boxes_after_filtered = [gb for gbi, gb in enumerate(gt_boxes) if target_items_after_filtered[gbi]]
    for i, box in enumerate(gt_boxes):
        cls_index = box['label'] if isinstance(box['label'], int) else _classes.index(box['label'])  # gt类别索引
        box_cls = _classes[cls_index]  # gt类别名称
        # 判别是否漏报
        if gt_status[i]:  # 非漏报场景: tpg
            update_items_recall[box_cls][cls_index] += 1
        elif gt_difficult[i]:  # 困难目标, 则不计算为漏报
            target_items_after_filtered[i] = False
        elif iou_matrix.shape[0] and iou_status_matrix[:, i].max():  # 误报场景: fp for gt
            # 选择最佳iou匹配
            pred_index = int(iou_status_matrix[:, i].argmax())
            pred_box = pred_boxes[pred_index]
            pred_cls_index = pred_box['label'] if isinstance(pred_box['label'], int) else _classes.index(pred_box['label'])
            update_items_recall[box_cls][pred_cls_index] += 1
        else:  # 漏报场景: fn
            update_items_recall[box_cls][-1] += 1

    # Note: 记录fp数据; 
    # fp有两种情况, 1. background预测为目标实例, 2. 目前类别预测其他类别, 且IOU大于阈值; 
    for j, box in enumerate(pred_boxes):
        cls_index = box['label'] if isinstance(box['label'], int) else _classes.index(box['label'])  # pred类别索引
        box_cls = _classes[cls_index]  # pred类别名称
        if pred_status[j]:  # 预测框命中
            update_items_precision[box_cls][cls_index] += 1
            continue
        
        # 判断是否和gt box通过匹配预值
        if iou_matrix[:, target_items_after_filtered].shape[1] and iou_status_matrix[j, target_items_after_filtered].max():  # 类别没有匹配, 但是IOU大于阈值
            # 获取iou_status_matrix[j, :]为True的索引
            gt_index = int(iou_status_matrix[j, target_items_after_filtered].argmax())
            gt_box = target_gt_boxes_after_filtered[gt_index]
            gt_cls_index = gt_box['label'] if isinstance(gt_box['label'], int) else _classes.index(gt_box['label'])
            # gt_cls = _classes[gt_cls_index]
            update_items_precision[box_cls][gt_cls_index] += 1
        else:  # 和GT关于IOU没有匹配上
            update_items_precision[box_cls][-1] += 1
    
    match_object["updateItemsRecall"] = update_items_recall
    match_object["updateItemsPrecision"] = update_items_precision
    return match_object


class StatisticBase:
    """统计基类

    实现了置信度配置、标签文件加载、标注内容转标注匹配格式、中间态标注内容钩子函数使用函数、
    置信度过滤钩子函数
    """

    def __init__(
            self, pred_suffix: Literal[".txt", ".json", ".xml"] = ".json",
            gt_suffix: Literal[".txt", ".json", ".xml"] = ".json",
            classes: Optional[Union[PathStr, List, Dict]] = None,
            chinese: Union[str, bool] = False, suffix_load_func: dict = {}, conf: Union[float, list] = 0, 
            ios_thresh: float = 0.5, iou_thresh: float = 0.5, **kwargs
        ):
        """
        :param Literal[.txt, .json, .xm] pred_suffix: 预测文件的后缀类型, defaults to ".json"
        :param Literal[.txt, .json, .xm] gt_suffix: 标注文件的后缀类型, defaults to ".json"
        :param dict suffix_load_func: 各类标签加载的方法, defaults to 空字典
        :param PathStr|list[str]|dict[str, Any] classes: 类别文件路径(支持list, dict), defaults to None
        :param float | list conf: 置信度阈值, defaults to 0
        :param float ios_thresh: 交自比阈值, defaults to 0.5
        :param float iou_thresh: 交并比阈值, defaults to 0.5
        """
        if classes is None:
            raise ValueError("classes参数不能为空")
        self.ios_thresh = ios_thresh
        self.iou_thresh = iou_thresh
        self.pred_suffix = pred_suffix
        self.gt_suffix = gt_suffix
        self.suffix_load_func = {".txt": read_yolo, ".json": parser_json, ".xml": read_voc}
        if suffix_load_func:
            self.suffix_load_func.update(suffix_load_func)
        self.classes = self.get_classes(deepcopy(classes))  # 直接处理形参，会有形参修改实参的副作用
        self.background = True  # 标记是否有背景类
        self.is_yolo_lbl = gt_suffix == ".txt" or pred_suffix == ".txt"
        if chinese:
            set_plt(font_path=chinese if isinstance(chinese, str) else None)
        self.conf = self.set_conf(conf)
    
    def set_conf(self, conf: Union[float, List]) -> Dict:
        """设置置信度阈值"""
        conf_dict = dict()
        if isinstance(conf, list):
            if len(conf) < len(self.classes):
                conf.extend([conf[-1]] * (len(self.classes) - len(conf)))
            conf_dict = {cls_name: c for cls_name, c in zip(self.classes, conf[:len(self.classes)])}
        elif isinstance(conf, float):
            conf_dict = {cls_name: conf for cls_name in self.classes}
        else:
            raise ValueError(f"不支持的置信度阈值类型{type(conf)}")
        return conf_dict

    @classmethod
    def get_classes(cls, class_file: Union[PathStr, List, Dict]) -> list:
        """获取类别列表

        :param class_file: 类别文件路径
        :type class_file: str
        """
        if isinstance(class_file, PathStr):
            classes = read_txt(str(class_file))
        elif isinstance(class_file, list):
            classes = class_file
        elif isinstance(class_file, dict):  # 传入是names时, 取name列表得到classes
            classes = [class_file[i] for i in sorted(class_file.keys())]
        else:
            raise ValueError(f"不支持的类别文件类型{type(class_file)}")
        if len(classes) == 0 or classes[-1] != 'background':
            classes.append('background')
        
        return classes

    def load_lbl_data(self, lbl_file: str, suffix: str) -> dict:
        """加载单个预测结果文件, 返回预测结果[dict]

        :param str lbl_file: 预测结果文件路径
        :param str suffix: 标签文件后缀名
        :return dict: 预测结果
        """
        lbl_data = self.suffix_load_func[suffix](lbl_file)
        return lbl_data

    @classmethod
    def labelme2match(cls, pred_entities: dict, **kwargs) -> list[dict]:
        """将labelme格式的对象转换为匹配格式, 匹配格式由自定义匹配模块定义

        :param dict pred_entities: labelme格式的预测结果对象
        :return list[dict]: 元素是labelme格式的预测对象, 形如: 
            shape={'label': 'car', 'bbox': None, 'polygon': [], 'flags': {'difficult': False}}
        """
        pred_boxes = []
        for shape in pred_entities['shapes']:
            box_cls = shape['label']
            x_list = [int(p[0]) for p in shape['points']]
            y_list = [int(p[1]) for p in shape['points']]
            x1, y1 = min(x_list), min(y_list)
            x2, y2 = max(x_list), max(y_list)
            bbox = [x1, y1, x2, y2]
            if not box_valid(bbox):
                logger.warning(f"预测框{bbox}无效, 跳过")
                if kwargs.get("verbose"):
                    print("shape数据展示为:")
                    pprint(shape)
                continue
            
            polygon = [[int(p[0]), int(p[1])] for p in shape['points']] if len(shape["points"]) > 4 else []
            conf = min(shape.get('conf', 1.0), shape.get("score", 1.0))
            difficult = bool(shape.get('difficult', False)) or bool(shape["flags"].get('difficult', False))
            pred_boxes.append(dict(
                label=box_cls, conf=conf, bbox=bbox, polygon=polygon,
                flags={"difficult": difficult}, shape_type=shape['shape_type'],
            ))
        
        return pred_boxes
    
    def yolo2match(self, gt_entities: list, **kwargs) -> list[dict]:
        """将标注文件加载内容转为匹配格式, 匹配格式由自定义匹配模块定义
        Note: 带conf的实例格式是: [class_id, x1, y1, x2, y2, conf]

        :param list gt_entities: 标注文件内容
        :return list[dict]: 元素是labelme格式的标注对象, 形如: 
            shape={'label': 'car', 'bbox': None, 'polygon': [], 'flags': {'difficult': False}}
        """

        img_h, img_w = kwargs.get('img_shape', (1, 1))
        gt_boxes = []
        for entity in gt_entities:
            
            label = self.classes[entity[0]]
            # 判断entity[1:]元素是否能被2整除, 若能则说明是带有conf的实例, 否则是不带conf的实例
            have_conf = True if len(entity[1:]) % 2 == 1 else False
            conf = entity[-1] if have_conf else 1.0
            end_index = len(entity) - 1 if have_conf else len(entity)
            bbox = []
            polygon = []
            if len(entity[1:end_index]) > 4:
                polygon = [
                    (
                        min(max(0, entity[2*i] * img_w), img_w), min(max(0, entity[2*i+1] * img_h), img_h)
                    ) for i in range(len(entity[1:end_index]) // 2)
                ]
                shape_type = "polygon"
            elif len(entity[1:end_index]) == 4:
                x, y, w, h = entity[1:5]
                x1, y1 = max(0, (x - w / 2) * img_w), max(0, (y - h / 2) * img_h)
                x2, y2 = min(img_w, (x + w / 2) * img_w), min(img_h, (y + h / 2) * img_h)
                bbox = [x1, y1, x2, y2]
                shape_type = "rectangle"
            else:
                logger.warning(f"标注框{entity[1:end_index]}无效, 跳过")
                continue

            gt_boxes.append(dict(
                label=label, conf=conf, bbox=bbox, polygon=polygon,
                flags={"difficult": False}, shape_type=shape_type,
            ))
        
        return gt_boxes
    
    @classmethod
    def voc2match(cls, gt_entities: dict, **kwargs) -> list[dict]:
        """将voc格式的标注文件加载内容转为匹配格式, 匹配格式由自定义匹配模块定义

        :param dict gt_entities: voc格式的标注文件内容
        :param bool use_conf: 是否使用置信度, defaults to False
        :return list[dict]: 元素是labelme格式的标注对象, 形如: 
            shape={'label': 'car', 'bbox': None, 'polygon': [], 'flags': {'difficult': False}}
        """
        result = []
        for obj in gt_entities["object"]:
            label = obj["name"]
            x1, y1 = obj["bndbox"]["xmin"], obj["bndbox"]["ymin"]
            x2, y2 = obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]
            if not box_valid((x1, y1, x2, y2)):
                logger.warning(f"标注框{x1, y1, x2, y2}无效, 跳过")
                continue
            conf = min(obj.get('conf', 1.0), obj.get('score', 1.0))
            difficult = bool(obj.get('difficult', 0))
                
            result.append({
                "label": label,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "polygon": [],
                "flags": {"difficult": difficult},
                "shape_type": "rectangle",  # Note: 目前只支持检测框, 多边形暂时不支持
            })
        
        return result
    
    def middle2match(self, entities: list | dict, suffix: str | None = None, **kwargs) -> list[dict]:
        """从labelme|yolo|voc格式的标注文件加载内容转为匹配格式, 匹配格式由自定义匹配模块定义

        :param list entities: 各种格式的标签直接加载的对象
        :param str suffix: 标签文件后缀名
        :return list[dict]: 示例列表
        """
        if entities is None:
            return []
        if suffix is None or suffix == ".json":
            assert isinstance(entities, dict), "json格式的标签文件内容必须是字典"
            return self.labelme2match(entities, **kwargs)
        elif suffix == ".txt":
            assert isinstance(entities, list), "txt格式的标签文件内容必须是列表"
            return self.yolo2match(entities, **kwargs)
        elif suffix == ".xml":
            assert isinstance(entities, dict), "xml格式的标签文件内容必须是字典"
            return self.voc2match(entities, **kwargs)
        else:
            raise ValueError(f"不支持的标签文件后缀名{suffix}")

    def get_image_shape(self, pred_entities, gt_entities, **kwargs) -> tuple:
        if not self.is_yolo_lbl:
            img_shape = (1, 1)
        if self.pred_suffix == ".json" and pred_entities:
            img_shape = (pred_entities['imageHeight'], pred_entities['imageWidth'])
        elif self.pred_suffix == ".xml" and pred_entities:
            img_shape = (pred_entities["size"]["height"], pred_entities["size"]["width"])
        elif self.gt_suffix == ".json" and gt_entities:
            img_shape = (gt_entities['imageHeight'], gt_entities['imageWidth'])
        elif self.gt_suffix == ".xml" and gt_entities:
            img_shape = (gt_entities["size"]["height"], gt_entities["size"]["width"])
        else:
            img_shape = (1, 1)
        return img_shape

    @classmethod
    def conf_filter(cls, label_list: list[dict], conf_thresh: dict, **kwargs) -> list[dict]:
        """根据置信度阈值过滤预测结果"""
        res_lbl = [lbl for lbl in label_list if lbl.get("conf", 1) >= conf_thresh[lbl["label"]]]
        return res_lbl

    def middle_post_process(self, label_list: list, call_backs: list, **kwargs) -> list:
        """标签归纳到中间态后的后处理, 调用回调函数对标签进行处理

        :param list label_list: 预测结果列表
        :param list call_backs: 回调函数列表
        :return list: 处理后的标签列表
        """
        for call_back in call_backs:
            label_list = call_back(label_list, **kwargs)
        return label_list

 
class StatisticConfusion(StatisticBase):
    """加载 PredictBase 推理结果 和 标签文件, 统计各类别的数量, 并保存到统计文件中
    推理结果文件夹和标注文件夹需要对应, 文件夹名称可以不一样.

    注: 全流程默认使用ultralytics的yolo模型, 若需要使用其他模型, 请重写以下方法:\n
        - gt2match : gt格式的对象转换为匹配格式, 匹配格式由自定义匹配模块定义
        - load_lbl_data : 读取标签文件, 返回标签对象, 目前只支持txt, json, xml格式
        - xxx2match: 各种数据格式转换为匹配格式, 匹配格式由自定义匹配模块定义
        - get_image_shape: 读取图片的shape, 用于YOLO标注box百分比坐标还原
        - match: 自定义匹配模块, 输入为gt和pred, 返回匹配结果

    Args:
        src_gt (list[str]): 标签文件路径, 支持多个数据子集，需要指定到数据子集的标签文件存放路径
        src_pred (list[str]): 推理结果文件路径, 支持多个数据子集，需要指定到数据子集的推理结果文件存放路径
        dst_dir (str): 预测结果保存目录
        project (str, optional): 实验名称, defaults to 'inference'.
        use_ios (bool, optional): 是否使用IOS计算, defaults to True
        classes (str, optional): 类别文件路径, defaults to 'classes.txt'.
        chinese (bool, optional): 是否使用中文类别, 可以指定中文字体文件的路径, defaults to True
        gt_suffix (str, optional): 标签文件后缀名, defaults to '.txt'.
        pred_suffix (str, optional): 推理结果文件后缀名, defaults to '.json'.
    
    注意: gt_suffix 和 pred_suffix 可以任选txt, json, xml格式, 但是若都使用YOLO格式, 则算法失效(不能得到像素坐标).
        
    Example:
        ```python
        >>> statistic = StatisticBase(
        ...     src_gt=['/data1/2024_datasets/infenrence/srcLabelTest-test/labels'],  # 标签文件路径
        ...     src_pred=[infer.dst_dir],  # 预测文件路径
        ...     dst_dir='/data1/2025_datasets/',  # 实验存放的根目录
        ...     project='statistic',  # 实验名称, 程序会默认追加 'Statistic'
        ...     use_ios=True,  # 是否使用IOS计算匹配程度
        ...     classes='/data1/classes.txt'  # 类别文件路径, YOLO格式
        ... )
        >>> statistic(
        ...     rendering=True,  # 是否渲染统计结果
        ...     ios_thresh=0.5,
        ...     iou_thresh=0.5
        ... )
        ```
    """

    def __init__(
            self, src_gt: list[PathStr], src_pred: list[PathStr], dst_dir: PathStr, 
            project: str = 'inference', use_ios: bool = True, 
            classes: Union[PathStr, List, Dict] = 'classes.txt', chinese: Union[bool, str] = True, 
            gt_suffix: Literal[".txt", ".json", ".xml"] = '.txt', 
            pred_suffix: Literal[".txt", ".json", ".xml"] = '.json', 
            use_fpfn: bool = False, conf: Union[float, List] = 0,
            ios_thresh: float = 0.5, iou_thresh: float = 0.5, 
            filter_category: list = [], difficult_filter: bool = True, **kwargs
        ):
        """初始化统计类

        :param list[PathStr] src_gt: 标签文件路径
        :param list[PathStr] src_pred: 推理结果文件路径
        :param PathStr dst_dir: 预测结果保存目录
        :param str project: 实验名称, defaults to 'inference'
        :param bool use_ios: 是否使用IOS计算, defaults to True
        :param Union[PathStr, List, Dict] classes: 类别文件路径, defaults to 'classes.txt'
        :param Union[bool, str] chinese: 是否使用中文类别, 可以指定中文字体文件的路径, defaults to True
        :param Literal[".txt", ".json", ".xml"] gt_suffix: 标签文件后缀名, defaults to '.txt'
        :param Literal[".txt", ".json", ".xml"] pred_suffix: 推理结果文件后缀名, defaults to '.json'
        :param bool use_fpfn: 是否使用FP, FN保存为子数据集, defaults to False
        :param float|list conf: 置信度阈值, 默认为0.001, 也可以传入一个列表, 对应每个类别的置信度阈值
        :param float ios_thresh: IOS阈值, defaults to 0.5
        :param float iou_thresh: IOU阈值, defaults to 0.5
        :param list filter_category: 过滤类别列表, defaults to []
        :param bool difficult_filter: 是否过滤difficult标记的标注, defaults to True
        """
        super().__init__(
            gt_suffix=gt_suffix, pred_suffix=pred_suffix, 
            suffix_load_func=kwargs.get('suffix_load_func', {}),
            classes=classes, chinese=chinese, conf=conf, 
            ios_thresh=ios_thresh, iou_thresh=iou_thresh, **kwargs
        )
        assert gt_suffix in self.suffix_load_func, f"不支持的标签文件后缀名{gt_suffix}"
        assert pred_suffix in self.suffix_load_func, f"不支持的预测文件后缀名{pred_suffix}"
        self.src_gt = path_list_valid(src_gt)
        self.src_pred = path_list_valid(src_pred)
        self.dst_dir = get_exp_dir(dst_dir, project + "_Statistic")
        self.project = project + "_Statistic"
        self.use_ios = use_ios
        self.background = False
        
        # 初始化统计的matrix
        self.matrix = ConfusionMatrix(
            len(self.classes), self.classes, 
            filter_category=filter_category, difficult_filter=difficult_filter
        )
        self.difficult_filter = difficult_filter
        self.use_fpfn = use_fpfn
        self.call_backs = [self.conf_filter]
        self.lock = threading.Lock()
    
    def load_datasets(self):
        """从预测文件加载数据集, 返回一个生成器对象, 第一个元素是items数量

        :yield: 预测文件, 标签文件元组
        """

        # 统计所有的预测结果文件, 没有预测预测文件也会保存
        datasets = self.src_pred
        
        # 统计预测文件数量
        total_num = 0
        for sub_datasets in datasets:
            if not sub_datasets.exists():
                raise ValueError(f"预测结果目录{sub_datasets}不存在")
            for file in sub_datasets.iterdir():
                if file.suffix != self.pred_suffix:
                    continue
                total_num += 1
        yield total_num

        for i, sub_datasets in enumerate(datasets):
            for file in sub_datasets.iterdir():
                if file.suffix != self.pred_suffix:
                    continue
                lbl_file = Path(self.src_gt[i]) / (file.stem + self.gt_suffix)
                yield (file, lbl_file), dict(ios_thresh=self.ios_thresh, iou_thresh=self.iou_thresh)

    def update(self, update_dict: dict, recall: bool = True):
        """更新统计矩阵

        :param update_dict: 更新字典, 格式为{类别: 预测向量}, 如: {"background": [1,0,0,1,0,1]}
        :type update_dict: dict
        """
        for key, value in update_dict.items():
            key_index = self.classes.index(key)
            with self.lock:
                if recall:
                    self.matrix.matrix_recall[:, key_index] += value     # 更新matrix_recall某一列
                else:
                    self.matrix.matrix_precision[key_index, :] += value  # 更新matrix_precision某一行

    def save_fpfn_pipeline(self, match_object: dict, gt_file: str, img_shape: tuple):
        """保存FP, FN实例为子数据集, 标签使用labelme格式
        :param match_object: 匹配结果
        :type match_object: dict
        :param gt_file: 标注文件路径
        :type gt_file: str
        :param img_shape: 图片shape
        :type img_shape: tuple

        note: match_object['boxesStatus']中保存了TPG, TPP, FP, FN四种实例列表, 实例的保存格式是: 
            shape={'label': 'car', 'bbox': None, 'polygon': [], 'flags': {'difficult': False}}
        """

        fp_list = match_object['boxesStatus']['fp']
        fn_list = match_object['boxesStatus']['fn']
        if len(fp_list) == 0 and len(fn_list) == 0:
            return None
        
        tpg_list = match_object['boxesStatus']['tpg']
        fnd_list = match_object['boxesStatus']['fnDiff']

        labelme_dict = {
            "version": "4.5.6",
            "flags": {
                "haveTp": True if len(tpg_list) > 0 else False,
                "haveFp": True if len(fp_list) > 0 else False,
                "haveFn": True if len(fn_list) > 0 else False,
            },
            "shapes": [],
            "imagePath": f"{Path(gt_file).stem}.jpg",
            "imageData": None,
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1],
        }

        def _add_instance(instance_list: list[dict], add_suffix: str = ""):
            for instance in instance_list:
                points = instance["polygon"] if instance["shape_type"] == "polygon" \
                    else [instance["bbox"][:2], instance["bbox"][2:]]
                labelme_dict['shapes'].append({
                    "label": f"{instance['label']}-{add_suffix}" if add_suffix else instance['label'],
                    "points": points,
                    "group_id": None,
                    "shape_type": instance["shape_type"],
                    "flags": {
                        # "predStatus": True if add_suffix != "fn" else False,
                        "difficult": instance.get('flags', {}).get('difficult', False),
                        "FP": add_suffix == "fp",
                        "FN": add_suffix == "fn",
                        "TPG": add_suffix == "tpg" or add_suffix == "",
                    }
                })
        
        _add_instance(tpg_list)
        _add_instance(fp_list, add_suffix='fp')
        _add_instance(fn_list, add_suffix='fn')
        _add_instance(fnd_list, add_suffix='fn')

        save_file_path = self.dst_dir / "fpfn_datasets" / f"{Path(gt_file).stem}.json"
        save_file_path.parent.mkdir(exist_ok=True, parents=True)
        save_labelme_label(save_file_path, labelme_dict)

    def match(self, pred_file: str, gt_file: str, **kwargs):
        pred_entities = self.load_lbl_data(pred_file, self.pred_suffix)
        gt_entities = self.load_lbl_data(gt_file, self.gt_suffix)
        img_shape = self.get_image_shape(pred_entities, gt_entities, **kwargs)
        # 匹配预测结果和标签文件
        pred_conf_boxes = self.middle2match(pred_entities, suffix=self.pred_suffix, img_shape=img_shape)
        gt_boxes = self.middle2match(gt_entities, suffix=self.gt_suffix, img_shape=img_shape)
        # 过滤中间态结果
        pred_boxes = self.middle_post_process(pred_conf_boxes, self.call_backs,  conf_thresh=self.conf, **kwargs)
        # 计算IoU
        ios_thresh = kwargs.get('ios_thresh', 0.5)
        iou_thresh = kwargs.get('iou_thresh', 0.5)
        match_object = obj_matcher(
            pred_boxes, gt_boxes, 
            use_ios=self.use_ios, mode="xyxy", 
            iou_thresh=iou_thresh, ios_thresh=ios_thresh,
            classes=self.classes
        )
        
        # 更新统计实验数据
        self.update(match_object['updateItemsRecall'], recall=True)
        self.update(match_object['updateItemsPrecision'], recall=False)
        self.matrix.update_img_wise_pr(match_object['boxesStatus'])

        # 判断是否需要对Difficult实例进行单独处理
        if self.difficult_filter:
            with self.lock:
                self.matrix.update_difficult_fn(match_object["boxesStatus"]["fnDiff"])
                self.matrix.update_difficult_tp(match_object["boxesStatus"]["tpDiff"])

        # 保存GT和误报实例为数据子集, 标签使用labelme格式
        if self.use_fpfn:
            self.save_fpfn_pipeline(match_object, gt_file, img_shape)

        return True

    def __call__(self, *args, **kwargs):
        """切片统计接口"""
        
        entities_generator = self.load_datasets()
        total_num = next(entities_generator)
        self.matrix.total_img_num = total_num  # type: ignore 第一个返回值是int类型

        # future开启多线程, 不能使用多进程（代码没有考虑多进程的情况）
        future_bar = FutureBar(total=total_num, desc="StatisticConfusion", max_workers=kwargs.get('max_workers', None))
        future_bar(self.match, entities_generator, total=total_num)
        # for entities in entities_generator:
        #     self.match(*entities[0], **entities[1])

        # 保存统计结果
        self.matrix.get_img_wise_eval()
        self.matrix.save_figure(dst_dir=self.dst_dir)
        self.matrix.save_xlsx(str(self.dst_dir / f"{self.dst_dir.name}_confusion_matrix.xlsx"))


class StatisticMatrix(StatisticBase):
    """预测分布指标统计
    
    :param list[str] pred_dir: 预测结果目录, defaults to None
    :param list[str] gt_dir: 标注文件目录, defaults to None
    :param str dst_dir: 结果保存目录, defaults to "."
    :param str project: 实验名称, defaults to "detection"
    :param bool plot: 是否绘制统计图, defaults to False
    :param list | dict | str names: 类别名称, defaults to {}
    :param str pred_suffix: 预测结果文件后缀名, defaults to ".json"
    :param str gt_suffix: 标注文件后缀名, defaults to ".json"
    tp (np.ndarray): True positive array.
    conf (np.ndarray): Confidence array.
    pred_cls (np.ndarray): Predicted class indices array.
    target_cls (np.ndarray): Target class indices array.
    on_plot (callable, optional): Function to call after plots are generated.

    """

    def __init__(
            self, pred_dir: list[str], gt_dir: list[str], 
            dst_dir=Path("."), project: str = "detection", plot=False,
            classes: list | dict | str = {},
            pred_suffix: Literal['.txt', '.json', '.xml']=".json",
            gt_suffix: Literal['.txt', '.json', '.xml']=".json",
            verbose=False, chinese=False, **kwargs
        ):
        super().__init__(
            pred_suffix=pred_suffix, gt_suffix=gt_suffix,
            classes=classes, chinese=chinese, conf=1e-6, **kwargs
        )
        assert pred_dir is not None and gt_dir is not None, "预测结果目录和标注文件目录不能为空"
        _pred_dir = pred_dir if isinstance(pred_dir, list) else [pred_dir]
        _gt_dir = gt_dir if isinstance(gt_dir, list) else [gt_dir]
        self.pred_dir: list[Path] = [Path(p) for p in _pred_dir]
        self.gt_dir: list[Path] = [Path(p) for p in _gt_dir]
        self.dst_dir = dst_dir if isinstance(dst_dir, Path) else Path(dst_dir)
        self.save_dir = get_exp_dir(self.dst_dir, project + "_Matrix")
        self.names = self.load_names(classes)
        self.nc = len(self.names)
        self.pred_suffix = pred_suffix
        self.gt_suffix = gt_suffix
        self.verbose = verbose
        self.seen = 0
        self.table = None
        
        self.matrix = DetMetrics(save_dir=self.save_dir, plot=plot, names=self.names, **kwargs)
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.iou_vector = np.linspace(0.5, 0.95, 10)  # IoU阈值向量
        self.num_iou = self.iou_vector.size  # IoU阈值数量
    
    @staticmethod
    def load_names(names: Union[list, dict, str]):
        if isinstance(names, PathStr):
            names_obj = read_txt(names)
        else:
            names_obj = names
        
        if isinstance(names_obj, list):
            names_dict = {i: v for i, v in enumerate(names_obj)}
        elif isinstance(names_obj, dict):
            names_dict = names_obj
        else:
            raise ValueError(f"不支持的names数据类型{type(names_obj)}")
        return names_dict
        
    def load_items(self):
        pred_num = 0
        for pred_dir in self.pred_dir:
            for pred_file in pred_dir.iterdir():
                if pred_file.suffix != self.pred_suffix:
                    continue
                pred_num += 1
        yield pred_num

        num = 0
        for pred_dir, gt_dir in zip(self.pred_dir, self.gt_dir):
            for pred_file in pred_dir.iterdir():
                if pred_file.suffix != self.pred_suffix:
                    continue
                gt_file = gt_dir / (pred_file.stem + self.gt_suffix)
                num += 1
                yield (pred_file, gt_file, num), dict()

    def match_cls_and_iou(self, pred_cls, gt_cls, iou_matrix):
        """根据IoU矩阵匹配预测结果和标签文件

        :param pred_cls: 预测框的预测类别数组
        :param gt_cls: 标注框的标签数组
        :param iou_matrix: IoU矩阵, 元素为[pred_idx, gt_idx, iou]
        """

        # 针对单张图片的预测与GT构建一个不同IOU阈值下的匹配记录矩阵
        correct = np.zeros((pred_cls.shape[0], self.iou_vector.shape[0])).astype(bool)
        # 记录预测与GT之间的类别匹配情况
        correct_cls = gt_cls[:, None] == pred_cls
        iou = iou_matrix * correct_cls

        for i, threshold in enumerate(self.iou_vector):
            matches = np.nonzero(iou > threshold)
            matches = np.array(matches).T  # 匹配上的索引, list of [pred_idx, gt_idx]
            if matches.shape[0]:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]  # 按照匹配的iou数值重排匹配数组
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]     # 去除重复的gt_idx
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]     # 去除重复的pred_idx
            correct[matches[:, 1].astype(int), i] = True                              # 标记每个预测对象在self.iou_vector[i]下是否有匹配
        
        return correct

    def match(self, pred_file: str, gt_file: str, num: int, **kwargs):
        
        pred_entities = self.load_lbl_data(pred_file, self.pred_suffix)
        gt_entities = self.load_lbl_data(gt_file, self.gt_suffix)
        img_shape = self.get_image_shape(pred_entities, gt_entities, **kwargs)
        # 匹配预测结果和标签文件
        ## pred_boxes的元素为: shape={'label': 'car', 'bbox': [], 'polygon': [], 'flags': {'difficult': False}}
        pred_middle = self.middle2match(pred_entities, suffix=self.pred_suffix, img_shape=img_shape, use_conf=True)
        gt_middle = self.middle2match(gt_entities, suffix=self.gt_suffix, img_shape=img_shape)
        pred_boxes = [[b['label'], b['conf'], *b['bbox']] for b in pred_middle]
        gt_boxes = [[b['label'], *b['bbox']] for b in gt_middle]
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 5))
        pred_boxes = np.array(pred_boxes) if pred_boxes else np.zeros((0, 6))

        stat = dict(
            conf=pred_boxes[:, 1].astype(float),  # 预测对象默认格式是: [label, conf, x1, y1, x2, y2]
            tp=np.zeros((pred_boxes.shape[0], self.num_iou), dtype=bool),
            pred_cls=pred_boxes[:, 0],
            target_cls=gt_boxes[:, 0],
            target_img=np.unique(gt_boxes[:, 0])
        )
        self.seen += 1

        # 没有预测时或着没有GT时, 立即整备self.stats, 并结束
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            for k in self.stats.keys():
                self.stats[k].append(stat[k])
            return None

        # 计算iou匹配矩阵
        iou_matrix = iou_np_boxes(gt_boxes[:, 1:], pred_boxes[:, 2:])
        iou_cls_iou_vector = self.match_cls_and_iou(pred_boxes[:, 0], gt_boxes[:, 0], iou_matrix)
        stat['tp'] = iou_cls_iou_vector

        for k in self.stats.keys():
            self.stats[k].append(stat[k])

        return None
    
    def bincount(self, x, minlength=1):
        if x.shape[0] and isinstance(x[0], np.str_):
            res_value, res_count = np.unique(x, return_counts=True)
            res = np.zeros(max(minlength, res_value.shape[0]), dtype=int)
            res_index = np.array([self.name2id[v] for v in res_value], dtype=int)
            res[res_index] = res_count
        else:
            res = np.bincount(x.astype(int), minlength=minlength)
        return res
    
    def get_stats(self):
        """返回分布指标统计结果"""
        self.name2id = {v: k for k, v in self.names.items()}
        stats = {k: np.concatenate(v) for k, v in self.stats.items()}
        self.nt_per_class = self.bincount(stats['target_cls'], minlength=self.nc)
        self.nt_per_image = self.bincount(stats['target_img'], minlength=self.nc)
        stats.pop('target_img', None)
        if len(stats) and stats["tp"].any():
            self.matrix.process(**stats)
        return self.matrix.results_dict
    
    @staticmethod
    def number_format(x):
        return f"{x:11.4g}"
    
    def print_results(self):
        """Prints training/validation set metrics per class."""
        table = PrettyTable()
        table.field_names = ["class", "images", "objects", *self.matrix.keys]
        table.add_row(["all", self.seen, self.nt_per_class.sum(), 
                       *[self.number_format(x) for x in self.matrix.mean_results()]])

        if self.nt_per_class.sum() == 0:
            logger.warning(f"WARNING ⚠️ no labels found, can not compute metrics without labels")

        # Print results per class
        if self.verbose and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.matrix.ap_class_index):
                table.add_row([
                    self.names[self.name2id[c]], 
                    self.nt_per_image[self.name2id[c]], 
                    self.nt_per_class[self.name2id[c]], 
                    *[self.number_format(x) for x in self.matrix.class_result(i)]
                ])
        print(table)
        self.table = table
    
    def __call__(self, *args, **kwargs):
        entities_generator = self.load_items()
        total_num = next(entities_generator)

        # 匹配预测结果和标签文件, 结果保存到self.match_matrix
        future_bar = FutureBar(total=total_num, desc="StatisticMatrix")
        future_bar(self.match, entities_generator, total=total_num)

        # 合并self.stats, 并计算指标
        self.get_stats()
        self.print_results()

        return self.matrix
