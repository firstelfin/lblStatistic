#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2026/05/08 21:47:49

import os
import sys
import warnings
import numpy as np
from numpy import ndarray
from pathlib import Path
from typing import List, Tuple, Union
warnings.filterwarnings('ignore')


def box_valid(box: Union[List, Tuple]) -> bool:
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return False
    return True


def xywh2xyxy(bbox: list) -> list:
    """xywh(cx, cy, w, h)矩形框标注转换为xyxy模式"""
    a, b, w, h = bbox
    w_shift = w // 2
    h_shift = h // 2
    a1, b1, a2, b2 = int(a-w_shift), int(b-h_shift), int(a+w_shift), int(b+h_shift)
    return [a1, b1, a2, b2]


def inter_box(box1: list, box2: list) -> tuple[list, float]:
    """_summary_

    :param list box1: 边框左上右下角坐标
    :param list box2: 边框左上右下角坐标
    :return tuple[list, float]: 交集的坐标和面积
    """
    # 计算交集区域的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if not box_valid([x_left, y_top, x_right, y_bottom]):
        area = 0
    else:
        area = (x_right - x_left) * (y_bottom - y_top)
    return [x_left, y_top, x_right, y_bottom], area


def ios_box(box1: list, box2: list, mode: str="xyxy", double: bool=False):
    """交自比

    :param list box1: 预测bbox
    :param list box2: 匹配的查询bbox
    :param str mode: bbox的组成模式, 'xywh'表示框中心和宽高, defaults to 'xyxy', options: ['xywh', 'xyxy']
    :param bool double: 是否双边计算, defaults to False
    """
    if mode not in ["xywh", "xyxy"]:
        raise Exception("modeError: IOP_box的mode参数不在可选范围内.")

    if mode == "xywh":
        bbox1 = xywh2xyxy(box1)
        bbox2 = xywh2xyxy(box2)
    else:
        bbox1 = box1
        bbox2 = box2
    
    # 求bbox的交集
    if not box_valid(bbox1) or not box_valid(bbox2):
        raise Exception(f"bboxError: 边框的坐标不符合要求, bbox1={bbox1}, bbox2={bbox2}.")
    
    _, inter_area = inter_box(bbox1, bbox2)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    ios1 = inter_area / bbox1_area
    ios2 = inter_area / bbox2_area
    if double:
        return ios1, ios2
    return max(ios1, ios2)


def iou_box(box1: list, box2: list, mode: str="xyxy", **kwargs) -> float:
    """计算两个边框的交并比

    :param list box1: x1, y1, x2, y2分别是左上和右下角坐标
    :param list box2: x1, y1, x2, y2分别是左上和右下角坐标
    :param str mode: bbox的组成模式, 'xywh'表示框中心和宽高, defaults to 'xyxy', options: ['xywh', 'xyxy']
    :raises Exception: box坐标不符合要求
    :return float: 交并比数值
    """
    if mode not in ["xywh", "xyxy"]:
        raise Exception("modeError: IOP_box的mode参数不在可选范围内.")

    if mode == "xywh":
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    if not box_valid(box1) or not box_valid(box2):
        raise Exception(f"bboxError: 边框的坐标不符合要求, bbox1={box1}, bbox2={box2}.")
    _, inter_area = inter_box(box1, box2)

    # 计算两个矩形框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集区域的面积
    union_area = box1_area + box2_area - inter_area

    # 计算交并比（IoU）
    iou = inter_area / union_area

    return iou


def iou_np_boxes(boxes1: ndarray, boxes2: ndarray, eps: float=1e-7) -> ndarray:
    """计算np数组中boxes1和boxes2的iou

    :param ndarray boxes1: 边框左上右下角坐标, instance = [x1, y1, x2, y2]
    :param ndarray boxes2: 边框左上右下角坐标
    :return ndarray: iou数组
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    (a1, a2) = np.split(np.expand_dims(boxes1.astype(float), axis=1), 2, axis=2)
    (b1, b2) = np.split(np.expand_dims(boxes2.astype(float), axis=0), 2, axis=2)
    inter = np.prod(np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None), axis=2)
    boxes1_area = np.prod(np.clip(a2 - a1, 0, None), axis=2)
    boxes2_area = np.prod(np.clip(b2 - b1, 0, None), axis=2)
    union = boxes1_area + boxes2_area - inter
    iou = inter / (union + eps)
    return iou


def path_list_valid(path_dir) -> List[Path]:
    if isinstance(path_dir, (str, Path)):
        datasets = [path_dir]
    else:
        datasets = path_dir
    datasets = [Path(dataset) for dataset in datasets]
    return datasets


def get_exp_dir(dst_dir: Union[str, Path], project: str = 'inference') -> Path:
    """获取实验结果保存目录, 文件夹不存在则创建

    :param str|Path dst_dir: 实验根目录
    :param str project: 实验名称, 默认为'inference'
    :return Path: 不重复的实验保存路径
    """
    res_dir = Path(dst_dir) / project
    if not res_dir.exists():
        res_dir.mkdir(exist_ok=True, parents=True)
        return res_dir
    elif not list(res_dir.iterdir()):
        return res_dir
    i = 1
    while (Path(dst_dir) / f'{project}{i}').exists():
        # 目录已存在, 判断时候为空文件夹
        if not (Path(dst_dir) / f'{project}{i}').iterdir():
            break
        i += 1
    (Path(dst_dir) / f'{project}{i}').mkdir(exist_ok=True, parents=True)
    return Path(dst_dir) / f'{project}{i}'
