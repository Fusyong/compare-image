"""
OpenCV工具函数
"""
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, cast

# 类型别名定义
ImageArray = NDArray[np.uint8]  # OpenCV图像类型

# OpenCV常量
IMREAD_COLOR = cv2.IMREAD_COLOR
COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
MARKER_CROSS = cv2.MARKER_CROSS
FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX

def cv_imread(filename: str) -> Optional[ImageArray]:
    """安全的图像读取函数"""
    img = cv2.imread(filename)
    if img is None:
        return None
    return cast(ImageArray, img)

def cv_resize(image: ImageArray, size: Tuple[int, int]) -> ImageArray:
    """安全的图像缩放函数"""
    return cast(ImageArray, cv2.resize(image, size))

def cv_cvtColor(image: ImageArray, code: int) -> ImageArray:
    """安全的颜色空间转换函数"""
    return cast(ImageArray, cv2.cvtColor(image, code))

def cv_drawMarker(
    image: ImageArray,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    markerType: int,
    markerSize: int,
    thickness: int
) -> None:
    """安全的标记点绘制函数"""
    cv2.drawMarker(image, position, color, markerType, markerSize, thickness)

def cv_putText(
    image: ImageArray,
    text: str,
    org: Tuple[int, int],
    fontFace: int,
    fontScale: float,
    color: Tuple[int, int, int],
    thickness: int
) -> None:
    """安全的文本绘制函数"""
    cv2.putText(image, text, org, fontFace, fontScale, color, thickness)