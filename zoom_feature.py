"""图像放大功能模块

提供图像局部放大功能，允许用户通过鼠标选择区域进行放大查看。
支持左右两侧图像的放大操作。

Classes:
    ZoomFeature: 图像放大功能类，处理图像选择和放大显示
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2  # type: ignore
from typing import Optional, Tuple, Any, cast
import numpy as np
from numpy.typing import NDArray

# 类型别名定义
ImageArray = NDArray[np.uint8]  # OpenCV图像类型
Point = Tuple[int, int]  # 点坐标类型 (x, y)

class ZoomFeature:
    """图像放大功能类

    提供图像局部放大功能，允许用户通过鼠标选择区域进行放大查看。
    支持左右两侧图像的放大操作。

    Attributes:
        app: 主应用实例
        selection_start: 选择区域的起始点坐标
        selection_end: 选择区域的结束点坐标
        selection_rect: 选择区域的矩形对象
        zoom_window: 放大窗口实例
        zoom_canvas: 放大画布实例
    """

    def __init__(self, app: Any) -> None:
        """初始化放大功能

        Args:
            app: 主应用实例
        """
        self.app = app
        self.selection_start: Optional[Point] = None
        self.selection_end: Optional[Point] = None
        self.selection_rect: Optional[int] = None
        self.zoom_window: Optional[tk.Toplevel] = None
        self.zoom_canvas: Optional[tk.Canvas] = None

        # 绑定事件
        self.app.left_canvas.bind("<Control-Button-1>", lambda e: self.start_selection(e, "left"))
        self.app.left_canvas.bind("<Control-B1-Motion>", lambda e: self.update_selection(e, "left"))
        self.app.left_canvas.bind("<Control-ButtonRelease-1>", lambda e: self.end_selection(e, "left"))

        self.app.right_canvas.bind("<Control-Button-1>", lambda e: self.start_selection(e, "right"))
        self.app.right_canvas.bind("<Control-B1-Motion>", lambda e: self.update_selection(e, "right"))
        self.app.right_canvas.bind("<Control-ButtonRelease-1>", lambda e: self.end_selection(e, "right"))

    def start_selection(self, event: tk.Event, side: str) -> None:
        """开始选择区域

        Args:
            event: 鼠标事件
            side: 图像侧（"left" 或 "right"）
        """
        canvas = self.app.left_canvas if side == "left" else self.app.right_canvas
        self.selection_start = (event.x, event.y)
        self.selection_rect = canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline='red', width=2
        )

    def update_selection(self, event: tk.Event, side: str) -> None:
        """更新选择区域

        Args:
            event: 鼠标事件
            side: 图像侧（"left" 或 "right"）
        """
        if self.selection_rect is None or self.selection_start is None:
            return

        canvas = self.app.left_canvas if side == "left" else self.app.right_canvas
        canvas.coords(
            self.selection_rect,
            self.selection_start[0], self.selection_start[1],
            event.x, event.y
        )

    def end_selection(self, event: tk.Event, side: str) -> None:
        """结束选择并显示放大图像

        Args:
            event: 鼠标事件
            side: 图像侧（"left" 或 "right"）
        """
        if self.selection_rect is None or self.selection_start is None:
            return

        canvas = self.app.left_canvas if side == "left" else self.app.right_canvas
        image = self.app.left_image if side == "left" else self.app.right_image

        if image is None:
            return

        # 获取选择区域的坐标
        x1, y1 = self.selection_start
        x2, y2 = event.x, event.y

        # 确保坐标是有效的
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # 计算图像上的实际位置
        img_height, img_width = image.shape[:2]
        x1_img = int(x1 * img_width / canvas_width)
        y1_img = int(y1 * img_height / canvas_height)
        x2_img = int(x2 * img_width / canvas_width)
        y2_img = int(y2 * img_height / canvas_height)

        # 确保坐标是有效的
        x1_img, x2_img = min(x1_img, x2_img), max(x1_img, x2_img)
        y1_img, y2_img = min(y1_img, y2_img), max(y1_img, y2_img)

        # 提取选择区域
        roi = image[y1_img:y2_img, x1_img:x2_img].copy()
        if roi.size == 0:
            return

        # 放大图像
        roi = cv2.resize(roi, None, fx=4, fy=4)  # type: ignore
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # type: ignore

        # 创建或更新放大窗口
        if self.zoom_window is None:
            self.zoom_window = tk.Toplevel(self.app.root)
            self.zoom_window.title("放大视图")
            self.zoom_window.protocol("WM_DELETE_WINDOW", self.clear_zoom)
            self.zoom_canvas = tk.Canvas(self.zoom_window)
            self.zoom_canvas.pack()
            self.zoom_canvas.bind("<Button-1>", lambda e: self.clear_zoom())

        # 更新放大窗口位置到屏幕中心
        screen_width = self.app.root.winfo_screenwidth()
        screen_height = self.app.root.winfo_screenheight()
        window_width = roi.shape[1]
        window_height = roi.shape[0]
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.zoom_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # 显示放大的图像
        photo = ImageTk.PhotoImage(image=Image.fromarray(roi))
        if self.zoom_canvas is not None:
            self.zoom_canvas.config(width=window_width, height=window_height)
            self.zoom_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            # 保存图片引用，防止被垃圾回收
            self.zoom_canvas._photo_ref = photo  # type: ignore

        # 清除选择矩形
        canvas.delete(self.selection_rect)
        self.selection_rect = None
        self.selection_start = None

    def clear_zoom(self) -> None:
        """清除放大窗口"""
        if self.zoom_window is not None:
            self.zoom_window.destroy()
            self.zoom_window = None
            self.zoom_canvas = None