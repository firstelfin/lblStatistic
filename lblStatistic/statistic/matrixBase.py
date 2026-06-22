#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   confusionMatrix.py
@Time    :   2024/08/26 17:38:44
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Optional, Union
from loguru import logger
from pathlib import Path
from xlsxwriter.utility import xl_col_to_name
from lblStatistic.utils import set_plt, PathStr
warnings.filterwarnings('ignore')


def array2picture(
        data, category, title_name: str = "Confusion Matrix", 
        dst_path: PathStr = '', mode=None, chinese: bool | str = False,
        x_label: str = "GT", y_label: str = "PREDICT",
        char_width_px: int = 8, char_height_px: int = 12,
        cell_padding: int = 12, label_padding: int = 18, 
        cmap: str = "viridis", change_last_axis_label= [None, None], **kwargs
    ):
    """将np.ndarray数据转换为图片并保存

    :param ndarray data: matrix数据, 可以是numpy二维数组, 支持float, int, str类型
    :param list category: 标签列表, 目前仅支持1-D列表
    :param str title_name: 图片标题, 默认为"Confusion Matrix"
    :param str dst_path: 图片保存路径, 默认为当前路径下保存为'Confusion Matrix.png', 后缀强制使用png
    :param str mode: 数值显示模式, 默认为None, 即自动判断
    :param bool chinese: 是否使用中文标签, 默认为False
    :param str x_label: 横坐标标签, 默认为"GT"
    :param str y_label: 纵坐标标签, 默认为"PREDICT"
    :param int char_width_px: 字符宽度（像素）
    :param int char_height_px: 字符高度（像素）
    :param int cell_padding: 单元格间距（像素）
    :param int label_padding: 标签间距（像素）
    :param str cmap: 颜色映射, 默认为"viridis"
    :param list change_last_axis_label: 是否修改最后一个标签, 默认为[None, None], 第一个元素为x轴标签, 第二个元素为y轴标签
    """

    if chinese:
        if isinstance(chinese, str):
            set_plt(font_path=chinese)
        elif isinstance(chinese, bool):
            set_plt()
        else:
            raise TypeError("chinese must be bool or str")

    # 0. 设置默认参数
    char_width_px = char_width_px
    grid_font_size = char_height_px
    cell_padding = cell_padding
    label_padding = label_padding
    dst_file_path = f"{title_name}.png" if not dst_path else Path(dst_path).with_suffix(".png")

    # 1. matrix数据计算最长长度
    fmt = "d" if mode is None else mode
    str_data = np.vectorize(lambda x: format(x, fmt))(data)
    max_str_len = max(max(len(s) for s in str_data.ravel()), 7)

    # 2. 计算单元格尺寸（像素）
    cell_width_px = max_str_len * char_width_px + cell_padding * 2
    cell_height_px = grid_font_size * 2 + cell_padding * 2

    # 3. 计算标签空间
    # 横纵坐标标签
    x_labels = deepcopy(category[:data.shape[0]])
    y_labels = deepcopy(category[:data.shape[1]])
    if change_last_axis_label[0] is not None:
        x_labels[-1] = change_last_axis_label[0]
    if change_last_axis_label[1] is not None:
        y_labels[-1] = change_last_axis_label[1]
    # 横纵坐标标签最大长度
    x_label_length = max(len(s) for s in x_labels)                         
    y_label_length = max(len(s) for s in y_labels)
    # 横纵坐标空间占比大小
    x_label_space = x_label_length * char_width_px + label_padding
    y_label_space = y_label_length * char_width_px + label_padding

    # 4. 计算标题空间
    title_fontsize = math.ceil(grid_font_size * 1.4)
    title_height_px = title_fontsize * 1.5
    x_axis_label_fontsize = math.ceil(grid_font_size * 1.1)
    y_axis_label_fontsize = math.ceil(grid_font_size * 1.1)

    # 5. 计算颜色条宽度
    colorbar_width_px = cell_width_px * 0.5

    # 6. 计算图像总像素尺寸（含标签、标题、颜色条）
    rows, cols = data.shape
    width_px = cols * cell_width_px + y_label_space + colorbar_width_px + y_axis_label_fontsize
    height_px = rows * cell_height_px + x_label_space + title_height_px + x_axis_label_fontsize
    # 7. 设置 DPI 和 图像尺寸（英寸）
    dpi = 100
    imgw = width_px / dpi
    imgh = height_px / dpi

    # 创建 figure（不使用 constrained_layout）
    fig, ax = plt.subplots(figsize=(imgw, imgh), dpi=dpi)

    # 设置颜色条位置（归一化坐标）
    cbar_left = 0.93
    cbar_bottom = (x_label_space + x_axis_label_fontsize)/ height_px
    cbar_width = 0.025
    cbar_height = (rows * cell_height_px - x_axis_label_fontsize) / height_px

    # 添加颜色条
    cbar_ax = fig.add_axes((cbar_left, cbar_bottom, cbar_width, cbar_height))

    # 绘制 heatmap
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap=cmap, xticklabels=y_labels, yticklabels=x_labels, ax=ax, 
        linewidths=1, annot_kws={"size": grid_font_size, "weight": "bold"}, cbar=True, cbar_ax=cbar_ax
    )

    # 添加标题
    ax.set_title(str(title_name), fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=y_axis_label_fontsize)
    ax.set_xlabel(x_label, fontsize=x_axis_label_fontsize)

    # 调整热图区域边距（为颜色条预留空间）
    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')  # 右对齐防止标签被截断

    # 使用 tight_layout 并指定热图区域范围（避免扩展到颜色条）
    plt.tight_layout(rect=(0, 0, 0.9, 1))

    # 保存图片
    plt.savefig(dst_file_path, dpi=dpi, bbox_inches='tight', format="png")


class ConfusionMatrix:
    r"""Confusion Matrix 绘制工具, 此对象接收预测值和真实值, 并根据预测值和真实值计算混淆矩阵, 并绘制混淆矩阵. 混淆矩阵记录
    GT类数据预测为PRED类数据的数量, 并可视化展示. 建议xtick尽量使用长度差异小的字符串.
    
    ### Note: 
        1. set_plt 用于设置matplotlib字体, 全局生效. 字体文件会自动下载, 并缓存到 /home/usename/.config/elfin/fonts/
        2. matrix 横纵坐标设置为yolo的样式, 降低理解成本
        3. 类别名称列表category可以为空, 则自动生成0-num_classes-1的索引列表, category长度应等于num_classes, 最后一个是backgroud

    Args:
        num_classes (int): 类别数量
        category (list[str], optional): 类别名称列表. Defaults to None.
        title (str, optional): 图标题. Defaults to "Confusion Matrix".
        cmap (str, optional): 颜色映射. Defaults to "YlGnBu".
        chinese (bool, optional): 是否使用简体中文. Defaults to False.
        exclude_zero (bool, optional): 是否排除0值. Defaults to True.
        filter_category (list[str], optional): 过滤掉的类别名称列表. Defaults to [].


    Attributes:
        num_classes (int): 类别数量
        matrix (np.ndarray): 混淆矩阵
        category (list[str]): 类别名称列表
        title (str): 图标题
        cmap (str): 颜色映射
        figure (matplotlib.figure.Figure): 绘图对象
        ax (matplotlib.axes.Axes): 绘图轴
    

    Examples:
        ```python
        >>> from codeUtils.matrix.confusionMatrix import ConfusionMatrix
        >>> cm = ConfusionMatrix(num_classes=3, category=["cat1", "cat2", "cat3"])
        >>> cm.add_matrix_item(i=0, j=1, value=1)
        >>> cm.add_matrix_item(i=1, j=2, value=2)
        >>> cm.add_matrix_item(i=0, j=2, value=3)
        >>> cm.save_figure(dst_dir="xxx/xxx")      # 保存混淆矩阵图片
        >>> cm.save_xlsx("confusionMatrix.xlsx")   # 保存混淆矩阵和Recall-Precision到Excel文件
        >>> cm.simplified_chinese_help()           # 简体中文支持帮助文档
        ```
    
    """

    class_name = "ConfusionMatrix"

    def __init__(
            self, num_classes, category: list[str], cmap: str = "YlGnBu", 
            chinese: bool | str = False, exclude_zero=True, filter_category=[],
            difficult_filter=False
        ):
        self.num_classes = num_classes
        self.matrix_recall = np.zeros((num_classes, num_classes), dtype=np.int32)
        self.matrix_precision = np.zeros((num_classes, num_classes), dtype=np.int32)
        self.normal_matrix_recall = None
        self.normal_matrix_precision = None
        self.difficult_fn = np.zeros(num_classes, dtype=np.int32)
        self.difficult_tp = np.zeros(num_classes, dtype=np.int32)
        self.category = category
        self.cmap = cmap
        if chinese:
            self.set_plt(font_path=chinese if isinstance(chinese, str) else None)
        self.exclude_zero = exclude_zero
        self.filter_category = np.array([False if filter_c in filter_category else True for filter_c in category])
        self.difficult_filter = difficult_filter
        self.imgwise_pr_recall = [0] * self.num_classes      # 召回图像数量记录
        self.imgwise_pr_fp = [0] * self.num_classes          # 图像误报数量记录
        # self.imgwise_pr_precision = [0] * self.num_classes   # 图像精确, 数量记录
        self.imgwise_gt_num = [0] * self.num_classes         # 图像GT数量记录
        self.total_img_num = 0

    @classmethod
    def set_plt(cls, font_path = None):
        set_plt(font_path=font_path)

    def get_img_wise_eval(self):
        self.img_wise_pr_recall = [self.imgwise_pr_recall[i] / max(self.imgwise_gt_num[i], 1) for i in range(self.num_classes)]
        self.img_wise_pr_precision = [
            self.imgwise_pr_recall[i] / max(self.imgwise_pr_fp[i]+self.imgwise_pr_recall[i], 1)
            for i in range(self.num_classes)
        ]
        self.img_wise_pr_accuracy = [max(self.total_img_num - self.imgwise_pr_fp[i], 0) / max(self.total_img_num, 1) for i in range(self.num_classes)]
    
    def update_difficult_fn(self, shapes):
        for shape in shapes:
            self.difficult_fn[self.category.index(shape["label"])] += 1
    
    def update_difficult_tp(self, shapes):
        for shape in shapes:
            self.difficult_tp[self.category.index(shape["label"])] += 1

    def update_img_wise_pr(self, update_dict: dict) -> str:
        """根据match的匹配结果, 统计每个类别图像级别的FP, TP, GT

        :param dict update_dict: _description_
        :return str: 整个图像级别的匹配状态, 包含TP, FP, ""(非缺陷)
        """

        self.total_img_num += 1

        # 统计GT数量
        gt_names = set()
        for shape in (update_dict["tpg"] + update_dict["fnDiff"] + update_dict["fn"]):
            gt_names.add(shape["label"])
        for name in gt_names:
            name_idx = self.category.index(name)
            self.imgwise_gt_num[name_idx] += 1
        if len(gt_names):
            self.imgwise_gt_num[-1] += 1

        # 先统计TP数据
        tp_names = set()
        for shape in (update_dict["tpg"] + update_dict["tpp"] + update_dict["tpDiff"]):
            tp_names.add(shape["label"])
        for name in tp_names:
            name_idx = self.category.index(name)
            self.imgwise_pr_recall[name_idx] += 1
        
        # 再统计FP数据, 排除TP数据
        fp_names = set()
        for shape in update_dict["fp"]:
            if shape["label"] in tp_names:
                continue
            fp_names.add(shape["label"])
        for name in fp_names:
            name_inx = self.category.index(name)
            self.imgwise_pr_fp[name_inx] += 1
        
        # 统计整个图像级别的数量
        res_status = ""
        if len(tp_names):
            res_status = "TP"
            self.imgwise_pr_recall[-1] += 1
        elif len(fp_names):
            res_status = "FP"
            self.imgwise_pr_fp[-1] += 1

        return res_status

    def add_matrix_item(self, i: int, j: int, value: int, recall: bool = True):
        if recall:
            self.matrix_recall[i][j] += value
        else:
            self.matrix_precision[i][j] += value

    def add_matrix_items(self, matrix: np.ndarray, recall: bool = True):
        if recall:
            self.matrix_recall += matrix
        else:
            self.matrix_precision += matrix

    def save_figure(self, dst_dir: str | Path):
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        self.get_normalize_matrix()

        # 绘制 self.matrix_recall
        array2picture(
            data=self.matrix_recall, category=self.category, title_name="Confusion Matrix Recall Num",
            dst_path=dst_dir / "confusionMatrix_recall.png", mode="d", cmap=self.cmap,
            change_last_axis_label=["FN", None]
        )
        # 绘制 self.matrix_precision
        array2picture(
            data=self.matrix_precision, category=self.category, title_name="Confusion Matrix Precision Num",
            dst_path=dst_dir / "confusionMatrix_precision.png", mode="d", cmap=self.cmap,
            change_last_axis_label=[None, "FP"]
        )
        # 绘制 self.normal_matrix_recall
        array2picture(
            data=self.normal_matrix_recall, category=self.category, title_name="Confusion Matrix Recall Rate",
            dst_path=dst_dir / "confusionMatrix_recall_rate.png", mode=".2f", cmap=self.cmap,
            change_last_axis_label=["FN", None]
        )
        # 绘制 self.normal_matrix_precision
        array2picture(
            data=self.normal_matrix_precision, category=self.category, title_name="Confusion Matrix Precision Rate",
            dst_path=dst_dir / "confusionMatrix_precision_rate.png", mode=".2f", cmap=self.cmap,
            change_last_axis_label=[None, "FP"]
        )

    @classmethod
    def simplified_chinese_help(cls):
        r"""简体中文帮助文档
        
        下载:
        ---------

            ```
            # step1: 下载 Source Han Serif SC 字体
            wget https://github.com/adobe-fonts/source-han-serif/releases/download/2.003R/09_SourceHanSerifSC.zip
            # step2: 解压
            unzip 09_SourceHanSerifSC.zip
            # step3: 复制到指定目录
            sudo cp -r 09_SourceHanSerifSC/SimplifiedChinese /usr/share/fonts/

            # step4: 刷新字体缓存
            sudo fc-cache -fv
            # step5:验证字体是否安装成功
            fc-list | grep "Source Han Serif SC"
            ```

        """
        print(cls.simplified_chinese_help.__doc__)

    @staticmethod
    def config_column_width(column_name):
        if "difficult" in column_name.lower():
            return 8
        else:
            return len(column_name)

    def save_xlsx(self, path: str):
        """保存混淆矩阵和Recall-Precision到Excel文件

        :param path: 文件路径
        :type path: str
        """
        eps = 1e-16
        self.get_normalize_matrix()  # 计算归一化的混淆矩阵

        pd_matrix_recall = pd.DataFrame(self.matrix_recall, columns=self.category, index=self.category)
        pd_matrix_precision = pd.DataFrame(self.matrix_precision, columns=self.category, index=self.category)
        pd_normal_matrix_recall = pd.DataFrame(self.normal_matrix_recall, columns=self.category, index=self.category)
        pd_normal_matrix_precision = pd.DataFrame(self.normal_matrix_precision, columns=self.category, index=self.category)

        gt_num = self.matrix_recall.sum(axis=0)
        pred_num = self.matrix_precision.sum(axis=1)
        gt_num[-1] = self.matrix_recall[self.filter_category][-1, :].sum()  # 计算漏报数量
        pred_num[-1] = self.matrix_precision[self.filter_category][:, -1].sum()  # 计算误报数量
        gt_num = gt_num[self.filter_category]
        pred_num = pred_num[self.filter_category]

        # 对角线元素
        recall_num = self.matrix_recall.diagonal()[self.filter_category]
        precision_num = self.matrix_precision.diagonal()[self.filter_category]
        fn_difficult_num = self.difficult_fn[self.filter_category]
        tp_difficult_num = self.difficult_tp[self.filter_category]
        difficult_num = fn_difficult_num + tp_difficult_num
        
        recall = np.round(recall_num / (gt_num + eps), decimals=8)
        precision = np.round(precision_num / (pred_num + eps), decimals=8)

        # 根据filter过滤无关类别(类别变少了)
        rp = np.stack([
            recall_num, gt_num, recall,              # 召回数量、GT数量、召回率
            precision_num, pred_num, precision,      # 精确数量、预测数量、精确率
            self.imgwise_pr_recall, self.imgwise_gt_num, self.img_wise_pr_recall,   # 图像级别召回数量、GT数量、召回率
            self.imgwise_pr_fp, self.img_wise_pr_precision, self.img_wise_pr_accuracy]   # 图像级别误报数量、精确率、准确率
            , axis=1)

        if self.difficult_filter:   # 添加difficult计数
            rp = np.hstack([rp, difficult_num[:, None], fn_difficult_num[:, None], tp_difficult_num[:, None]])

        # 整体召回精度计算需要排除background, 默认索引为-1
        gt_total_num = gt_num[:-1].sum()
        pred_total_num = pred_num[:-1].sum()
        recall_total_num = recall_num[:-1].sum()
        precision_total_num = precision_num[:-1].sum()
        total_recall = (np.sum(recall_num[:-1]) + eps) / (gt_total_num + eps)
        total_precision = (np.sum(precision_num[:-1]) + eps) / (pred_total_num + eps)
        total_recall = np.round(total_recall, decimals=8)
        total_precision = np.round(total_precision, decimals=8)
        difficult_extend = [""] * 3 if self.difficult_filter else []
        # 添加总计行
        total_new_row = [
            recall_total_num, gt_total_num, total_recall,
            precision_total_num, pred_total_num, total_precision,
            self.imgwise_pr_recall[-1], self.imgwise_gt_num[-1], self.img_wise_pr_recall[-1],              # 图像级别召回数量、GT数量、召回率
            self.imgwise_pr_fp[-1], self.img_wise_pr_precision[-1], self.img_wise_pr_accuracy[-1]   # 图像级别误报数量、精确率、准确率
        ]+ difficult_extend
        rp = np.vstack([rp, total_new_row])
        rp[:, [8, 10, 11]] = np.round(rp[:, [8, 10, 11]].astype(float), decimals=8)
        rp[-2, 0], rp[-2, 3] = "FN", "FP"
        rp[-2, 6:12] = ["FN", self.imgwise_gt_num[-1] - self.imgwise_pr_recall[-1], "-", self.imgwise_pr_fp[-1], "-", "-"]

        # 预测计数为0, GT计数为0的类别, 在exclude_zero模式下排除
        if self.exclude_zero:
            index_array = np.bitwise_or(gt_num[:-1] != 0, pred_num[:-1] != 0)
        else:
            index_array = np.ones_like(recall[:-1], dtype=bool)

        # 计算平均值
        mr = np.round(np.mean(recall[:-1][index_array]), decimals=8)
        mp = np.round(np.mean(precision[:-1][index_array]), decimals=8)
        category_img_man_recall = np.round(np.mean(np.array(self.img_wise_pr_recall[:-1])[index_array]), decimals=8)
        category_img_man_precision = np.round(np.mean(np.array(self.img_wise_pr_precision[:-1])[index_array]), decimals=8)
        category_img_man_acc = np.round(np.mean(np.array(self.img_wise_pr_accuracy[:-1])[index_array]), decimals=8)
        mean_new_row = [
            "-", "-", mr, "-", "-", mp,
            "-", "-", category_img_man_recall,
            "-", category_img_man_precision, category_img_man_acc
        ] + difficult_extend
        rp = np.vstack([rp, mean_new_row])
        
        _rp_index = [self.category[i] for i in range(len(self.category)) if self.filter_category[i]] + ["Total", "Mean"]
        _rp_columns = [
            "召回量", "标注量", "召回率", "正确预测量", "预测量", "精度",
            "Img召回量", "Img标注量", "Img召回率", "Img误报量", "Img精确率", "Img准确率"
        ]
        if self.difficult_filter:
            _rp_columns += ["DifficultNum", "DifficultFN", "DifficultTP"]
        df_rp = pd.DataFrame(rp, columns=_rp_columns, index=_rp_index)
        
        for col in df_rp.columns:
            # 尝试转换为 float 或 int，无法转换的保持原样
            df_rp[col] = pd.to_numeric(df_rp[col], errors='coerce').fillna(df_rp[col])
            # is_numeric = pd.to_numeric(df_rp[col], errors='coerce').notna()
            # df_rp[col][is_numeric] = pd.to_numeric(df_rp[col][is_numeric], errors='coerce')
        
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            # 设置表头和内容样式
            start_row, start_col = (2, 2)
            workbook = writer.book
            # ✅ 设置默认字体大小（模拟 Excel 默认）
            DEFAULT_FONT_SIZE = 10
            workbook.formats[0].set_font_size(DEFAULT_FONT_SIZE)
            # 设置表头和内容样式
            format_header = workbook.add_format({
                'bold': True,
                'align': 'center',
                'font_size': DEFAULT_FONT_SIZE + 1,  # 假设正文是 10，这里加粗字体 11
                'top': 2,         # 上边框
                'bottom': 1,      # 下边框
            })
            format_bottom = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'center',      # 居中对齐
                'bottom': 2,      # 只在底部下方画线
            })
            format_content = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'center',      # 居中对齐
            })
            format_index = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'left'
            })
            format_index_bottom = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'left',
                'bottom': 2
            })
            summary_format = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'center',
                'valign': 'vcenter',
                'text_wrap': True,
                'top': 3,
                'bottom': 2,
            })

            # 写入混淆矩阵到xlsx文件
            pd_matrix_recall.to_excel(writer, sheet_name="Confusion Matrix Recall Num")
            pd_matrix_precision.to_excel(writer, sheet_name="Confusion Matrix Precision Num")
            pd_normal_matrix_recall.to_excel(writer, sheet_name="Confusion Matrix Recall Rate")
            pd_normal_matrix_precision.to_excel(writer, sheet_name="Confusion Matrix Precision Rate")

            # 创建一个空的sheet
            worksheet_rp = workbook.add_worksheet("Recall-Precision")
            
            # 写入索引（index）
            last_row = 0
            for row_num, data in enumerate(df_rp.index):
                worksheet_rp.write(row_num + start_row + 1, start_col, data, format_index)
                last_row = row_num
            # 写入列名（columns）
            worksheet_rp.write(start_row , start_col, None, format_header)
            for col_num, data in enumerate(df_rp.columns):
                worksheet_rp.write(start_row , start_col + 1 + col_num, data, format_header)
            # 写入数据
            for row_num, row_data in enumerate(df_rp.values):
                for col_num, data in enumerate(row_data):
                    worksheet_rp.write(row_num + start_row + 1, start_col + 1 + col_num, data, format_content)
            # 设置表格底部的线
            worksheet_rp.write(last_row + start_row + 1, start_col, df_rp.index[-1], format_index_bottom)
            for col_num, data in enumerate(df_rp.values[last_row]):
                worksheet_rp.write(last_row + start_row + 1, col_num + start_col + 1, data, format_bottom)
            # 自动调整列宽
            for idx, col in enumerate(df_rp):  # Iterate through data to auto fit
                series = df_rp[col]
                max_len = max((
                    series.astype(str).map(len).max(), 10, # len of largest item
                    self.config_column_width(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
                worksheet_rp.set_column(idx + 1 + start_col, idx + 1 + start_col, max_len)  # set column width
            # 自动调整索引列的列宽
            index_len = max(df_rp.index.astype(str).map(len).max(), 5)  # 5 is the minimum width
            worksheet_rp.set_column(start_col, start_col, index_len)  # set index column width
            
            # 写入统计信息到xlsx文件
            if self.difficult_filter:
                total_fn_difficult_num = fn_difficult_num.sum()
                total_gt_num = total_fn_difficult_num + gt_total_num
                difficult_statistic = f"总计图像{self.total_img_num}个, GT实例{total_gt_num}个, 上报实例{pred_total_num}个\n" + \
                    f"困难实例{difficult_num.sum()}个, 发现{tp_difficult_num.sum()}个, 遗漏{total_fn_difficult_num}个"
                merge_start_cell = f"{xl_col_to_name(start_col+rp.shape[1]-2)}{start_row+rp.shape[0]}"  # 行编码从1开始,且表头和索引分别要占一行一列
                merge_end_cell = f"{xl_col_to_name(start_col+rp.shape[1])}{start_row+rp.shape[0]+1}"
                merge_cell_range = f"{merge_start_cell}:{merge_end_cell}"
                worksheet_rp.merge_range(merge_cell_range, difficult_statistic, summary_format)  # type: ignore

    def get_normalize_matrix(self):
        """获取归一化的混淆矩阵"""
        if self.normal_matrix_recall is None:
            self.normal_matrix_recall = np.zeros_like(self.matrix_recall)
            gt_num = self.matrix_recall.sum(axis=0, keepdims=True).clip(min=1)
            self.normal_matrix_recall = 100 *self.matrix_recall / gt_num

        if self.normal_matrix_precision is None:
            self.normal_matrix_precision = np.zeros_like(self.matrix_precision)
            pred_num = self.matrix_precision.sum(axis=1, keepdims=True).clip(min=1)
            self.normal_matrix_precision = 100 * self.matrix_precision / pred_num
