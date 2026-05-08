#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2026/05/08 21:34:02

from pathlib import Path
from typing import Optional
from matplotlib import rcParams
from matplotlib import font_manager


def valid_local_font(font_path: Optional[str] = None):
    """加载本地字体文件，并设置 matplotlib 全局字体

    :param str font_path: 字体文件路径, defaults to None
    """
    chinese_font = ["SimHei", "Arial.Unicode", "PingFang"]
    if font_path is not None and Path(font_path).exists():
        return font_path
    else:
        for name in chinese_font:
            temp_path = Path.home() / f'.config/elfin/fonts/{name}.ttf'
            if temp_path.exists():
                return str(temp_path)
        print(f"字体文件 {font_path} 不存在, 也未发现中文字体文件. 下载请调用命令 'lblConvert font --download' 下载默认字体文件！🏇")
        return None


def set_plt(font_path: Optional[str] = None):
    if font_path is None or not Path(font_path).exists():
        font_dir = Path.home() / ".config/elfin/fonts/"
        if not font_dir.exists():
            raise FileNotFoundError(f"字体文件目录 {font_path if font_path else font_dir} 不存在！")
        font_path = valid_local_font(font_path)
        if font_path is None:
            raise FileNotFoundError(f"未找到有效的中文字体文件！")
    font_prop = font_manager.FontProperties(fname=font_path)
    # 获取字体名称
    font_name = font_prop.get_name()
    font_manager.fontManager.addfont(font_path)
    # 更新 rcParams 设置
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [font_name]  # 替换为实际字体名称
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
