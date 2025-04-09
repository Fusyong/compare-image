"""
图像比对工具
"""
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog
from typing import List, Tuple, Optional, Dict, Any, cast, TYPE_CHECKING

from PIL import Image, ImageTk
import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

from image_processor import ImageProcessor

# 类型别名定义
Coordinates = Tuple[float, float]  # 坐标点 (x, y)
ImageArray = NDArray[np.uint8]  # OpenCV图像类型
MarkerList = List[Coordinates]  # 标记点列表
CacheItem = Dict[str, Any]  # 缓存项类型

# OpenCV常量
if TYPE_CHECKING:
    IMREAD_COLOR: int
    COLOR_BGR2RGB: int
    MARKER_CROSS: int
    FONT_HERSHEY_SIMPLEX: int
else:
    IMREAD_COLOR = cv2.IMREAD_COLOR
    COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    MARKER_CROSS = cv2.MARKER_CROSS
    FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX

def cv_imread(filename: str) -> Optional[ImageArray]:
    """安全的图像读取函数

    Args:
        filename: 图像文件路径

    Returns:
        读取的图像数据，如果读取失败则返回None
    """
    img = cv2.imread(filename)  # type: ignore
    if img is None:
        return None
    return cast(ImageArray, img)

def cv_resize(image: ImageArray, size: Tuple[int, int]) -> ImageArray:
    """安全的图像缩放函数

    Args:
        image: 输入图像
        size: 目标尺寸 (width, height)

    Returns:
        缩放后的图像
    """
    return cast(ImageArray, cv2.resize(image, size))  # type: ignore

def cv_cvtColor(image: ImageArray, code: int) -> ImageArray:
    """安全的颜色空间转换函数

    Args:
        image: 输入图像
        code: 转换代码

    Returns:
        转换后的图像
    """
    return cast(ImageArray, cv2.cvtColor(image, code))  # type: ignore

def cv_drawMarker(
    image: ImageArray,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    markerType: int,
    markerSize: int,
    thickness: int
) -> None:
    """安全的标记点绘制函数

    Args:
        image: 输入图像
        position: 标记点位置 (x, y)
        color: 颜色 (B,G,R)
        markerType: 标记类型
        markerSize: 标记大小
        thickness: 线条粗细
    """
    cv2.drawMarker(image, position, color, markerType, markerSize, thickness)  # type: ignore

def cv_putText(
    image: ImageArray,
    text: str,
    org: Tuple[int, int],
    fontFace: int,
    fontScale: float,
    color: Tuple[int, int, int],
    thickness: int
) -> None:
    """安全的文本绘制函数

    Args:
        image: 输入图像
        text: 要绘制的文本
        org: 文本位置 (x, y)
        fontFace: 字体
        fontScale: 字体大小
        color: 颜色 (B,G,R)
        thickness: 线条粗细
    """
    cv2.putText(image, text, org, fontFace, fontScale, color, thickness)  # type: ignore

class ImageComparisonApp:
    """图像比较应用主类

    负责处理图像加载、显示、比较和用户交互等功能。

    Attributes:
        root (tk.Tk): 主窗口实例
        processor (ImageProcessor): 图像处理器实例
        left_image (Optional[ImageArray]): 左侧图像数据
        right_image (Optional[ImageArray]): 右侧图像数据
        left_markers (MarkerList): 左图标记点列表
        right_markers (MarkerList): 右图标记点列表
        left_images (List[str]): 左图文件路径列表
        right_images (List[str]): 右图文件路径列表
        current_index (int): 当前显示的图片索引
        is_comparing (bool): 是否处于比较状态
        comparison_cache (List[CacheItem]): 比较结果缓存
        active_marker (Optional[Tuple[str, int]]): 当前活动的标记点，格式为(side, index)
    """

    def __init__(self, root: tk.Tk) -> None:
        """初始化图像比较应用

        Args:
            root: Tkinter主窗口实例
        """
        self.root = root
        self.root.title("图像比对工具")

        # 设置窗口默认最大化
        self.root.state('zoomed')

        # 初始化图像处理器
        self.processor = ImageProcessor()
        self.processor.set_info_callback(self.show_info)

        # 图像数据
        self.left_image: Optional[ImageArray] = None
        self.right_image: Optional[ImageArray] = None
        self.left_markers: MarkerList = [(5.0, 5.0), (95.0, 95.0)]  # L1, L2
        self.right_markers: MarkerList = [(5.0, 5.0), (95.0, 95.0)]  # R1, R2
        self.active_marker: Optional[Tuple[str, int]] = None  # (side, index)
        self.left_result: Optional[ImageArray] = None
        self.right_result: Optional[ImageArray] = None

        # 图片组数据
        self.left_images: List[str] = []
        self.right_images: List[str] = []
        self.current_index: int = 0

        # 放大镜参数
        self.magnifier_size: int = 100  # 放大区域的大小
        self.magnifier_scale: int = 2   # 放大倍数改为2倍
        self.magnifier_visible: bool = False
        self.last_mouse_x: int = 0
        self.last_mouse_y: int = 0

        # 添加区域选择相关属性
        self.selection_start: Optional[Tuple[int, int]] = None
        self.selection_end = None
        self.selection_rect: Optional[int] = None
        self.zoom_window: Optional[tk.Toplevel] = None
        self.zoom_canvas: Optional[tk.Canvas] = None

        # 比较状态
        self.is_comparing: bool = False
        self.compare_mode: str = "compare"  # "compare" 或 "overlay" 或 "ocr"

        # 比较结果缓存
        self.comparison_cache: List[CacheItem] = []
        self.cache_index: int = -1

        self.setup_ui()
        self.bind_shortcuts()    # 绑定快捷键

        # 在窗口尺寸确定后自动加载图片
        self.root.after(100, self.auto_load_images)

    def auto_load_images(self):
        """自动加载图片"""
        left_dir = "img/L"
        right_dir = "img/R"

        if os.path.exists(left_dir) and os.path.exists(right_dir):
            # 获取目录下的所有图片文件，支持png和jpg两种格式
            left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))
            right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))
            left_images.extend(sorted(glob.glob(os.path.join(left_dir, "*.jpg"))))
            right_images.extend(sorted(glob.glob(os.path.join(right_dir, "*.jpg"))))

            if left_images and right_images:
                self.left_images = left_images
                self.right_images = right_images
                self.current_index = 0
                self.load_current_images()
                self.show_info(f"已自动加载第{self.current_index + 1}对图片")
            else:
                self.show_info("未找到PNG图片")
        else:
            self.show_info("未找到img/L或img/R目录")

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建左右分栏
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 工具栏（移到左侧）
        toolbar = ttk.LabelFrame(left_panel, text="操作")
        toolbar.pack(fill=tk.X, pady=5)

        # 创建一个Frame来容纳两个按钮
        buttons_frame = ttk.Frame(toolbar)
        buttons_frame.pack(fill=tk.X, padx=5, pady=2)

        # 添加加载图片组按钮
        ttk.Button(buttons_frame, text="加载左图组", command=lambda: self.load_image_group("left")).pack(side=tk.LEFT, expand=True, padx=(0,2))
        ttk.Button(buttons_frame, text="加载右图组", command=lambda: self.load_image_group("right")).pack(side=tk.LEFT, expand=True, padx=(2,0))

        # 添加图片导航信息
        nav_frame = ttk.Frame(toolbar)
        nav_frame.pack(fill=tk.X, padx=5, pady=2)
        self.nav_label = ttk.Label(nav_frame, text="")
        self.nav_label.pack()

        # 坐标输入区域 - 移动到这里，放在按钮下方
        coords_frame = ttk.LabelFrame(toolbar, text="校准点坐标")
        coords_frame.pack(fill=tk.X, padx=5, pady=2)

        # 左图坐标
        left_coords = ttk.Frame(coords_frame)
        left_coords.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(left_coords, text="左图:").pack()

        l1_frame = ttk.Frame(left_coords)
        l1_frame.pack(fill=tk.X)
        ttk.Label(l1_frame, text="L1: ").pack(side=tk.LEFT)
        self.l1_x = ttk.Entry(l1_frame, width=5)
        self.l1_x.pack(side=tk.LEFT)
        self.l1_y = ttk.Entry(l1_frame, width=5)
        self.l1_y.pack(side=tk.LEFT)
        # 绑定回车键事件
        self.l1_x.bind("<Return>", lambda e: self.update_marker_coordinates("left", 0))
        self.l1_y.bind("<Return>", lambda e: self.update_marker_coordinates("left", 0))

        l2_frame = ttk.Frame(left_coords)
        l2_frame.pack(fill=tk.X)
        ttk.Label(l2_frame, text="L2: ").pack(side=tk.LEFT)
        self.l2_x = ttk.Entry(l2_frame, width=5)
        self.l2_x.pack(side=tk.LEFT)
        self.l2_y = ttk.Entry(l2_frame, width=5)
        self.l2_y.pack(side=tk.LEFT)
        # 绑定回车键事件
        self.l2_x.bind("<Return>", lambda e: self.update_marker_coordinates("left", 1))
        self.l2_y.bind("<Return>", lambda e: self.update_marker_coordinates("left", 1))

        # 右图坐标
        right_coords = ttk.Frame(coords_frame)
        right_coords.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(right_coords, text="右图:").pack()

        r1_frame = ttk.Frame(right_coords)
        r1_frame.pack(fill=tk.X)
        ttk.Label(r1_frame, text="R1: ").pack(side=tk.LEFT)
        self.r1_x = ttk.Entry(r1_frame, width=5)
        self.r1_x.pack(side=tk.LEFT)
        self.r1_y = ttk.Entry(r1_frame, width=5)
        self.r1_y.pack(side=tk.LEFT)
        # 绑定回车键事件
        self.r1_x.bind("<Return>", lambda e: self.update_marker_coordinates("right", 0))
        self.r1_y.bind("<Return>", lambda e: self.update_marker_coordinates("right", 0))

        r2_frame = ttk.Frame(right_coords)
        r2_frame.pack(fill=tk.X)
        ttk.Label(r2_frame, text="R2: ").pack(side=tk.LEFT)
        self.r2_x = ttk.Entry(r2_frame, width=5)
        self.r2_x.pack(side=tk.LEFT)
        self.r2_y = ttk.Entry(r2_frame, width=5)
        self.r2_y.pack(side=tk.LEFT)
        # 绑定回车键事件
        self.r2_x.bind("<Return>", lambda e: self.update_marker_coordinates("right", 1))
        self.r2_y.bind("<Return>", lambda e: self.update_marker_coordinates("right", 1))

        # 添加模式选择 - 移到坐标输入区域后面
        mode_frame = ttk.LabelFrame(toolbar, text="比较模式")
        mode_frame.pack(fill=tk.X, padx=5, pady=2)

        # 模式选择 - 第一行
        mode_select_frame = ttk.Frame(mode_frame)
        mode_select_frame.pack(fill=tk.X, padx=5, pady=2)
        self.mode_var = tk.StringVar(value="compare")
        ttk.Radiobutton(mode_select_frame, text="像素", variable=self.mode_var, value="compare").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_select_frame, text="叠加", variable=self.mode_var, value="overlay").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_select_frame, text="OCR", variable=self.mode_var, value="ocr").pack(side=tk.LEFT, padx=5)

        # 叠加模式的透明度控制 - 第二行
        alpha_frame = ttk.Frame(mode_frame)
        alpha_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(alpha_frame, text="叠加透明度:").pack(side=tk.LEFT)
        self.alpha_entry = ttk.Entry(alpha_frame, width=5)
        self.alpha_entry.insert(0, "50")  # 默认透明度50%
        self.alpha_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(alpha_frame, text="%").pack(side=tk.LEFT)
        # 绑定回车键事件
        self.alpha_entry.bind("<Return>", self.on_alpha_change)

        # 比较按钮
        self.compare_button = ttk.Button(toolbar, text="比较/原图 Ctrl+P", command=self.toggle_compare)
        self.compare_button.pack(fill=tk.X, padx=5, pady=2)

        # 添加翻页按钮
        page_frame = ttk.Frame(toolbar)
        page_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(page_frame, text="上页 PgUp", command=lambda: self.navigate_images(-1)).pack(side=tk.LEFT, expand=True, padx=(0,2))
        ttk.Button(page_frame, text="下页 PgDn", command=lambda: self.navigate_images(1)).pack(side=tk.LEFT, expand=True, padx=(2,0))

        # 添加提示信息区域到工具栏
        info_frame = ttk.LabelFrame(toolbar, text="提示信息")
        info_frame.pack(fill=tk.X, padx=5, pady=2)
        self.info_label = ttk.Label(info_frame, text="", wraplength=170)
        self.info_label.pack(fill=tk.X, padx=5, pady=2)

        # 添加放大器按钮
        self.zoom_mode = False
        self.zoom_button = ttk.Button(toolbar, text="放大器", command=self.toggle_zoom_mode)
        self.zoom_button.pack(fill=tk.X, padx=5, pady=2)

        # 图像显示区域（在右侧面板）
        image_frame = ttk.Frame(right_panel)
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.left_canvas = tk.Canvas(image_frame, bg='gray')
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_canvas = tk.Canvas(image_frame, bg='gray')
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 绑定事件
        self.left_canvas.bind("<Button-1>", lambda e: self.on_left_click(e))
        self.left_canvas.bind("<B1-Motion>", lambda e: self.on_left_drag(e))
        self.left_canvas.bind("<ButtonRelease-1>", lambda e: self.on_left_release(e))

        self.right_canvas.bind("<Button-1>", lambda e: self.on_right_click(e))
        self.right_canvas.bind("<B1-Motion>", lambda e: self.on_right_drag(e))
        self.right_canvas.bind("<ButtonRelease-1>", lambda e: self.on_right_release(e))

    def show_info(self, message):
        """显示提示信息"""
        self.info_label.config(text=message)
        self.root.update()

    def check_and_use_cache(self) -> bool:
        """检查是否有可用的缓存，如果有则使用

        Returns:
            bool: 是否使用了缓存
        """
        if self.left_image is None or self.right_image is None:
            return False

        # 检查是否有当前图片对的缓存
        current_cache = None
        for cache in self.comparison_cache:
            # 检查图片路径和比较模式是否匹配
            if (cache.get("left_path") == self.left_images[self.current_index] and
                cache.get("right_path") == self.right_images[self.current_index] and
                cache.get("mode") == self.mode_var.get()):
                current_cache = cache
                break

        if current_cache is not None:
            # 检查图片和坐标是否发生变化
            left_img = current_cache.get("original_left")
            right_img = current_cache.get("original_right")
            left_markers = current_cache.get("left_markers")
            right_markers = current_cache.get("right_markers")

            if (left_img is not None and right_img is not None and
                left_markers is not None and right_markers is not None and
                isinstance(left_img, np.ndarray) and isinstance(right_img, np.ndarray) and
                np.array_equal(self.left_image, left_img) and
                np.array_equal(self.right_image, right_img) and
                self.left_markers == left_markers and
                self.right_markers == right_markers):

                print("\n=== 使用缓存的比较结果 ===")
                print(f"比较模式: {self.mode_var.get()}")
                print(f"左图坐标: L1({self.left_markers[0][0]:.1f}, {self.left_markers[0][1]:.1f}), "
                      f"L2({self.left_markers[1][0]:.1f}, {self.left_markers[1][1]:.1f})")
                print(f"右图坐标: R1({self.right_markers[0][0]:.1f}, {self.right_markers[0][1]:.1f}), "
                      f"R2({self.right_markers[1][0]:.1f}, {self.right_markers[1][1]:.1f})")

                # 使用缓存的比较结果
                left_result = current_cache.get("left")
                right_result = current_cache.get("right")
                if left_result is not None and self.left_canvas:
                    self.display_image(self.left_canvas, left_result, self.left_markers)
                if right_result is not None and self.right_canvas:
                    self.display_image(self.right_canvas, right_result, self.right_markers)

                self.show_info("使用缓存的比较结果")
                return True
            else:
                print("\n=== 缓存无效 ===")
                if not (isinstance(left_img, np.ndarray) and isinstance(right_img, np.ndarray) and
                       np.array_equal(self.left_image, left_img) and
                       np.array_equal(self.right_image, right_img)):
                    print("原因：图片内容已变化")
                if self.left_markers != left_markers or self.right_markers != right_markers:
                    print("原因：标记点已变化")
                # 清除当前缓存
                self.comparison_cache = [cache for cache in self.comparison_cache if not (
                    cache.get("left_path") == self.left_images[self.current_index] and
                    cache.get("right_path") == self.right_images[self.current_index]
                )]

        print("\n=== 需要重新比较 ===")
        return False

    def toggle_compare(self):
        """切换比较状态"""
        if self.left_image is None or self.right_image is None:
            self.show_info("请先加载左右两张图片")
            return

        if not self.is_comparing:
            self.is_comparing = True
            self.compare_button.configure(text="原图 Ctrl+P")

            # 检查是否有可用的缓存
            if not self.check_and_use_cache():
                # 显示处理中的提示
                if self.mode_var.get() == "ocr":
                    self.show_info("正在OCR识别...")
                else:
                    self.show_info("正在处理...")
                # 开始比较
                self.start_comparison()
        else:
            # 切换回原图模式
            self.is_comparing = False
            self.compare_button.configure(text="比较 Ctrl+P")
            self.show_info("")  # 清空提示信息

            # 显示原图
            if self.left_image is not None:
                self.display_image(self.left_canvas, self.left_image, self.left_markers)
            if self.right_image is not None:
                self.display_image(self.right_canvas, self.right_image, self.right_markers)

    def on_alpha_change(self, event):
        """透明度变化时的回调函数"""
        try:
            alpha = int(self.alpha_entry.get())
            if 0 <= alpha <= 100:
                if self.mode_var.get() == "overlay" and self.is_comparing:
                    # 如果当前是叠加模式且正在比较，重新开始比较
                    self.start_comparison()
                    self.show_info(f"叠加透明度已调整为: {alpha}%")
            else:
                self.show_info("透明度必须在0-100之间")
                self.alpha_entry.delete(0, tk.END)
                self.alpha_entry.insert(0, "50")
        except ValueError:
            self.show_info("请输入有效的数字")
            self.alpha_entry.delete(0, tk.END)
            self.alpha_entry.insert(0, "50")

    def start_comparison(self) -> None:
        """开始图像比较

        根据当前选择的比较模式（像素、叠加、OCR）执行图像比较。
        比较结果会被缓存，并在界面上显示。
        """
        try:
            print("\n=== 开始新的比较 ===")
            mode = self.mode_var.get()
            print(f"比较模式: {mode}")

            # 获取最新的校准坐标
            self.save_current_coordinates()

            # 检查图像和坐标的有效性
            if self.left_image is None or self.right_image is None:
                print("错误：图像未加载")
                self.show_info("请先加载两张图片")
                return

            if not self.left_markers or not self.right_markers:
                print("错误：标记点无效")
                self.show_info("请先设置有效的标记点")
                return

            # 打印详细的比较信息
            print("=== 比较参数 ===")
            print(f"左图尺寸: {self.left_image.shape}")
            print(f"右图尺寸: {self.right_image.shape}")
            print(f"左图标记点: L1({self.left_markers[0][0]:.1f}, {self.left_markers[0][1]:.1f}), "
                  f"L2({self.left_markers[1][0]:.1f}, {self.left_markers[1][1]:.1f})")
            print(f"右图标记点: R1({self.right_markers[0][0]:.1f}, {self.right_markers[0][1]:.1f}), "
                  f"R2({self.right_markers[1][0]:.1f}, {self.right_markers[1][1]:.1f})")

            # 执行比较
            print("\n=== 执行比较 ===")
            if mode == "compare":
                print("使用像素比较模式")
                self.left_result = self.processor.compare_images(
                    self.left_image, self.right_image,
                    self.left_markers, self.right_markers,
                    "left"
                )
                self.right_result = self.processor.compare_images(
                    self.left_image, self.right_image,
                    self.left_markers, self.right_markers,
                    "right"
                )
            elif mode == "overlay":
                print("使用叠加比较模式")
                try:
                    alpha = float(self.alpha_entry.get()) / 100.0 if self.alpha_entry else 0.5
                    print(f"叠加透明度: {alpha:.2f}")
                except ValueError:
                    alpha = 0.5
                    print(f"使用默认透明度: {alpha:.2f}")
                self.left_result = self.processor.overlay_images(
                    self.left_image, self.right_image,
                    self.left_markers, self.right_markers,
                    "left", alpha
                )
                self.right_result = self.processor.overlay_images(
                    self.left_image, self.right_image,
                    self.left_markers, self.right_markers,
                    "right", alpha
                )
            else:  # ocr mode
                print("使用OCR比较模式")
                self.left_result = self.processor.ocr_and_compare(
                    self.left_image, self.right_image,
                    self.left_markers, self.right_markers,
                    "left"
                )
                self.right_result = self.processor.ocr_and_compare(
                    self.left_image, self.right_image,
                    self.left_markers, self.right_markers,
                    "right"
                )

            # 检查比较结果
            print("\n=== 比较结果 ===")
            if self.left_result is None or self.right_result is None:
                print("错误：比较结果为空")
                self.show_info("比较失败，请检查图像和标记点")
                return

            # 缓存比较结果
            print("缓存比较结果")
            cache_item: CacheItem = {
                "left": self.left_result.copy() if self.left_result is not None else None,
                "right": self.right_result.copy() if self.right_result is not None else None,
                "mode": mode,
                "left_path": self.left_images[self.current_index],
                "right_path": self.right_images[self.current_index],
                "original_left": self.left_image.copy(),
                "original_right": self.right_image.copy(),
                "left_markers": self.left_markers.copy(),
                "right_markers": self.right_markers.copy()
            }
            self.comparison_cache.append(cache_item)

            # 限制缓存数量
            while len(self.comparison_cache) > 3:
                for i, cache in enumerate(self.comparison_cache):
                    if not (
                        cache.get("left_path") == self.left_images[self.current_index] and
                        cache.get("right_path") == self.right_images[self.current_index]
                    ):
                        self.comparison_cache.pop(i)
                        break

            self.cache_index = -1

            # 显示结果
            print("显示比较结果")
            if self.left_result is not None and self.left_canvas:
                self.display_image(self.left_canvas, self.left_result, self.left_markers)
            if self.right_result is not None and self.right_canvas:
                self.display_image(self.right_canvas, self.right_result, self.right_markers)

            # 更新提示信息
            if mode == "ocr":
                if np.array_equal(self.left_result, self.left_image if mode == "left" else self.right_image):
                    self.show_info("OCR结果相同")
                else:
                    self.show_info("已标记出差异区域并生成HTML报告")
            else:
                self.show_info("比较完成")

            print("=== 比较完成 ===\n")

        except Exception as e:
            import traceback
            print("\n=== 比较出错 ===")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("错误堆栈:")
            traceback.print_exc()
            self.show_info(f"比较出错: {type(e).__name__} - {str(e)}")
            self.is_comparing = False
            if self.compare_button:
                self.compare_button.config(text="比较 Ctrl+P")

    def load_image_group(self, side):
        """加载图片组"""
        dir_path = filedialog.askdirectory(title=f"选择{side}图组目录")
        if not dir_path:
            return

        # 获取目录下的所有图片文件
        image_files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
        if not image_files:
            self.show_info(f"在{dir_path}中未找到PNG图片")
            return

        if side == "left":
            self.left_images = image_files
            self.show_info(f"已加载左图组，共{len(image_files)}张图片")
        else:
            self.right_images = image_files
            self.show_info(f"已加载右图组，共{len(image_files)}张图片")

        # 如果两边都有图片，加载第一对
        if self.left_images and self.right_images:
            self.current_index = 0
            self.load_current_images()
            self.show_info(f"已加载第{self.current_index + 1}对图片")

    def load_current_images(self):
        """加载当前索引位置的图片"""
        # 保存当前坐标
        self.save_current_coordinates()

        if 0 <= self.current_index < len(self.left_images):
            self.load_image("left", self.left_images[self.current_index])
        if 0 <= self.current_index < len(self.right_images):
            self.load_image("right", self.right_images[self.current_index])

        # 恢复保存的坐标
        self.restore_coordinates()

        # 更新导航信息
        if self.left_images and self.right_images:
            self.nav_label.config(text=f"当前图片: {self.current_index + 1}/{min(len(self.left_images), len(self.right_images))}")

    def save_current_coordinates(self):
        """保存当前坐标到文件（像素值）"""
        if self.left_image is None or self.right_image is None:
            return

        # 获取当前图片文件名
        left_filename = os.path.basename(self.left_images[self.current_index])
        right_filename = os.path.basename(self.right_images[self.current_index])

        # 确保coords目录存在
        coords_dir = "coords"
        if not os.path.exists(coords_dir):
            try:
                os.makedirs(coords_dir)
                print(f"创建目录: {coords_dir}")
            except Exception as e:
                print(f"创建目录失败: {str(e)}")
                return

        # 创建坐标文件
        coord_file = os.path.join(coords_dir, f"{left_filename}.txt")

        try:
            # 保存坐标（像素值）
            with open(coord_file, 'w') as f:
                f.write(f"L1: {int(self.left_markers[0][0])} {int(self.left_markers[0][1])}\n")
                f.write(f"L2: {int(self.left_markers[1][0])} {int(self.left_markers[1][1])}\n")
                f.write(f"R1: {int(self.right_markers[0][0])} {int(self.right_markers[0][1])}\n")
                f.write(f"R2: {int(self.right_markers[1][0])} {int(self.right_markers[1][1])}\n")

            print(f"已保存坐标到文件: {coord_file}")
            print(f"左图坐标: L1({int(self.left_markers[0][0])}, {int(self.left_markers[0][1])}), L2({int(self.left_markers[1][0])}, {int(self.left_markers[1][1])})")
            print(f"右图坐标: R1({int(self.right_markers[0][0])}, {int(self.right_markers[0][1])}), R2({int(self.right_markers[1][0])}, {int(self.right_markers[1][1])})")
            print("注意：坐标值为像素值")
        except Exception as e:
            print(f"保存坐标文件失败: {str(e)}")
            return

    def restore_coordinates(self):
        """恢复保存的坐标值"""
        self.update_coordinate_entries()

    def bind_shortcuts(self):
        """绑定快捷键"""
        self.root.bind("<Prior>", lambda e: self.navigate_images(-1))  # PgUp
        self.root.bind("<Next>", lambda e: self.navigate_images(1))    # PgDn
        self.root.bind("<Control-p>", lambda e: self.toggle_compare())  # Ctrl+P
        self.root.bind("<Control-Left>", lambda e: self.view_cache(-1))  # Ctrl+Left
        self.root.bind("<Control-Right>", lambda e: self.view_cache(1))  # Ctrl+Right

    def navigate_images(self, direction):
        """导航到上一张或下一张图片"""
        if not self.left_images or not self.right_images:
            return

        new_index = self.current_index + direction
        if 0 <= new_index < min(len(self.left_images), len(self.right_images)):
            # 更新索引
            self.current_index = new_index

            # 加载新图片
            self.load_current_images()

            # 如果当前处于比较状态，开始比较
            if self.is_comparing:
                # 先检查是否有可用的缓存
                if not self.check_and_use_cache():
                    self.start_comparison()

    def load_image(self, side: str, file_path: str) -> None:
        """加载图片

        Args:
            side: 图片加载位置（"left" 或 "right"）
            file_path: 图片文件路径
        """
        try:
            image = cv_imread(file_path)
            if image is None:
                raise ValueError("无法加载图片")

            if side == "left":
                self.left_image = image
                if self.left_canvas:
                    self.display_image(self.left_canvas, image, self.left_markers)
            else:
                self.right_image = image
                if self.right_canvas:
                    self.display_image(self.right_canvas, image, self.right_markers)

            self.update_coordinate_entries()
            self.show_info(f"已加载{side}图: {os.path.basename(file_path)}")
        except Exception as e:
            self.show_info(f"加载图片失败: {str(e)}")

    def display_image(self, canvas: tk.Canvas, image: ImageArray, markers: MarkerList) -> None:
        """在画布上显示图像和标记点

        Args:
            canvas: 目标画布
            image: 要显示的图像数据
            markers: 要显示的标记点列表
        """
        # 获取画布尺寸
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
            canvas.configure(width=canvas_width, height=canvas_height)

        # 计算缩放比例
        img_height, img_width = image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # 调整图像大小
        resized_image = cv_resize(image, (new_width, new_height))
        rgb_image = cv_cvtColor(resized_image, COLOR_BGR2RGB)

        # 绘制标记点
        for i, (x, y) in enumerate(markers):
            px = int(x * new_width / 100)
            py = int(y * new_height / 100)
            cv_drawMarker(rgb_image, (px, py), (255, 0, 0), MARKER_CROSS, 20, 2)
            cv_putText(rgb_image,
                     f"{'L' if canvas == self.left_canvas else 'R'}{i+1}",
                     (px+5, py-5),
                     FONT_HERSHEY_SIMPLEX,
                     0.5, (255, 0, 0), 1)

        # 显示图像
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_image))
        canvas.delete("all")  # 清除画布
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        # 使用setattr来避免类型检查错误
        setattr(canvas, "_photo_ref", photo)

    def on_left_click(self, event):
        """处理左图点击事件"""
        if self.zoom_mode:
            self.start_selection(event, "left")
        else:
            self.start_drag(event, "left")

    def on_left_drag(self, event):
        """处理左图拖动事件"""
        if self.zoom_mode:
            self.update_selection(event, "left")
        else:
            self.drag(event, "left")

    def on_left_release(self, event):
        """处理左图释放事件"""
        if self.zoom_mode:
            self.end_selection(event, "left")
        else:
            self.end_drag(event)

    def on_right_click(self, event):
        """处理右图点击事件"""
        if self.zoom_mode:
            self.start_selection(event, "right")
        else:
            self.start_drag(event, "right")

    def on_right_drag(self, event):
        """处理右图拖动事件"""
        if self.zoom_mode:
            self.update_selection(event, "right")
        else:
            self.drag(event, "right")

    def on_right_release(self, event):
        """处理右图释放事件"""
        if self.zoom_mode:
            self.end_selection(event, "right")
        else:
            self.end_drag(event)

    def start_drag(self, event: tk.Event, side: str) -> None:
        """开始拖动标记点

        Args:
            event: 鼠标事件
            side: 图像侧（"left" 或 "right"）
        """
        canvas = self.left_canvas if side == "left" else self.right_canvas
        markers = self.left_markers if side == "left" else self.right_markers

        # 根据比较状态选择正确的图像
        if self.is_comparing:
            image = self.left_result if side == "left" else self.right_result
        else:
            image = self.left_image if side == "left" else self.right_image

        # 检查是否点击了标记点
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        x_rel = event.x / canvas_width * 100
        y_rel = event.y / canvas_height * 100

        for i, (mx, my) in enumerate(markers):
            if abs(x_rel - mx) < 5 and abs(y_rel - my) < 5:
                self.active_marker = (side, i)
                self.magnifier_visible = True
                self.last_mouse_x = event.x
                self.last_mouse_y = event.y
                if image is not None:
                    self.update_magnifier(event, canvas, image)
                break

    def drag(self, event: tk.Event, side: str) -> None:
        """处理拖动事件

        Args:
            event: 鼠标事件
            side: 图像侧（"left" 或 "right"）
        """
        if not self.active_marker or self.active_marker[0] != side:
            return

        canvas = self.left_canvas if side == "left" else self.right_canvas
        markers = self.left_markers if side == "left" else self.right_markers

        # 根据比较状态选择正确的图像
        if self.is_comparing:
            image = self.left_result if side == "left" else self.right_result
        else:
            image = self.left_image if side == "left" else self.right_image

        # 更新标记点位置
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        x_rel = min(max(event.x / canvas_width * 100, 0), 100)
        y_rel = min(max(event.y / canvas_height * 100, 0), 100)

        markers[self.active_marker[1]] = (x_rel, y_rel)

        # 更新显示
        if side == "left" and image is not None:
            self.display_image(self.left_canvas, image, self.left_markers)
        elif side == "right" and image is not None:
            self.display_image(self.right_canvas, image, self.right_markers)

        # 更新放大镜
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        if image is not None:
            self.update_magnifier(event, canvas, image)

        # 更新坐标输入框
        self.update_coordinate_entries()

    def end_drag(self, event):
        self.active_marker = None
        self.magnifier_visible = False
        # 清除放大镜
        if hasattr(self, 'magnifier_window'):
            self.magnifier_window.destroy()
            delattr(self, 'magnifier_window')

    def update_coordinate_entries(self) -> None:
        """更新坐标输入框的值（像素值）"""
        if not all([self.l1_x, self.l1_y, self.l2_x, self.l2_y,
                   self.r1_x, self.r1_y, self.r2_x, self.r2_y]):
            return

        # 更新左图坐标
        for entry, value in [
            (self.l1_x, self.left_markers[0][0]),
            (self.l1_y, self.left_markers[0][1]),
            (self.l2_x, self.left_markers[1][0]),
            (self.l2_y, self.left_markers[1][1])
        ]:
            entry.delete(0, tk.END)
            entry.insert(0, str(int(value)))

        # 更新右图坐标
        for entry, value in [
            (self.r1_x, self.right_markers[0][0]),
            (self.r1_y, self.right_markers[0][1]),
            (self.r2_x, self.right_markers[1][0]),
            (self.r2_y, self.right_markers[1][1])
        ]:
            entry.delete(0, tk.END)
            entry.insert(0, str(int(value)))

        # 打印当前坐标
        print(f"当前坐标 - 左图: L1({int(self.left_markers[0][0])}, {int(self.left_markers[0][1])}), "
              f"L2({int(self.left_markers[1][0])}, {int(self.left_markers[1][1])})")
        print(f"当前坐标 - 右图: R1({int(self.right_markers[0][0])}, {int(self.right_markers[0][1])}), R2({int(self.right_markers[1][0])}, {int(self.right_markers[1][1])})")
        self.l1_x.delete(0, tk.END)
        self.l1_x.insert(0, str(int(self.left_markers[0][0])))
        self.l1_y.delete(0, tk.END)
        self.l1_y.insert(0, str(int(self.left_markers[0][1])))

        self.l2_x.delete(0, tk.END)
        self.l2_x.insert(0, str(int(self.left_markers[1][0])))
        self.l2_y.delete(0, tk.END)
        self.l2_y.insert(0, str(int(self.left_markers[1][1])))

        # 更新右图坐标
        self.r1_x.delete(0, tk.END)
        self.r1_x.insert(0, str(int(self.right_markers[0][0])))
        self.r1_y.delete(0, tk.END)
        self.r1_y.insert(0, str(int(self.right_markers[0][1])))

        self.r2_x.delete(0, tk.END)
        self.r2_x.insert(0, str(int(self.right_markers[1][0])))
        self.r2_y.delete(0, tk.END)
        self.r2_y.insert(0, str(int(self.right_markers[1][1])))

        # 打印当前坐标
        print(f"当前坐标 - 左图: L1({int(self.left_markers[0][0])}, {int(self.left_markers[0][1])}), L2({int(self.left_markers[1][0])}, {int(self.left_markers[1][1])})")
        print(f"当前坐标 - 右图: R1({int(self.right_markers[0][0])}, {int(self.right_markers[0][1])}), R2({int(self.right_markers[1][0])}, {int(self.right_markers[1][1])})")
        print("注意：坐标值为像素值")

    def update_magnifier(self, event: tk.Event, canvas: tk.Canvas, image: ImageArray) -> None:
        """更新放大镜显示

        Args:
            event: 鼠标事件
            canvas: 目标画布
            image: 要显示的图像（原图或比较结果）
        """
        if not self.magnifier_visible or image is None:
            return

        # 获取鼠标位置对应的图像坐标
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        img_height, img_width = image.shape[:2]

        # 计算图像上的实际位置
        x = int(event.x * img_width / canvas_width)
        y = int(event.y * img_height / canvas_height)

        # 确定放大区域的范围
        half_size = self.magnifier_size // (2 * self.magnifier_scale)
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(img_width, x + half_size)
        y2 = min(img_height, y + half_size)

        # 提取放大区域
        roi = image[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return

        # 放大图像
        magnified_size = (self.magnifier_size, self.magnifier_size)
        roi = cv_resize(roi, magnified_size)
        roi = cv_cvtColor(roi, COLOR_BGR2RGB)

        # 在放大的图像中心绘制十字线
        center = self.magnifier_size // 2
        cv_drawMarker(roi, (center, center), (255, 0, 0), MARKER_CROSS, 20, 1)

        # 创建或更新放大镜窗口
        if not hasattr(self, 'magnifier_window') or not self.magnifier_window:
            self.magnifier_window = tk.Toplevel(self.root)
            self.magnifier_window.overrideredirect(True)
            self.magnifier_canvas = tk.Canvas(self.magnifier_window,
                                           width=self.magnifier_size,
                                           height=self.magnifier_size)
            self.magnifier_canvas.pack()

        # 更新放大镜位置
        screen_x = self.root.winfo_rootx() + canvas.winfo_x() + event.x
        screen_y = self.root.winfo_rooty() + canvas.winfo_y() + event.y
        if self.magnifier_window:
            self.magnifier_window.geometry(f"{self.magnifier_size}x{self.magnifier_size}+{screen_x+20}+{screen_y+20}")

            # 显示放大的图像
            if self.magnifier_canvas:
                photo = ImageTk.PhotoImage(image=Image.fromarray(roi))
                self.magnifier_canvas.delete("all")
                self.magnifier_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                setattr(self.magnifier_canvas, "_photo_ref", photo)

    def view_cache(self, direction):
        """查看缓存的比较结果"""
        if not self.comparison_cache:
            return

        # 计算新的缓存索引
        new_index = self.cache_index + direction
        if new_index < -1:  # -1表示不查看缓存
            new_index = -1
        elif new_index >= len(self.comparison_cache):
            new_index = len(self.comparison_cache) - 1

        self.cache_index = new_index

        if self.cache_index == -1:
            # 显示当前图片
            if self.left_image is not None:
                self.display_image(self.left_canvas, self.left_image, self.left_markers)
            if self.right_image is not None:
                self.display_image(self.right_canvas, self.right_image, self.right_markers)
            self.show_info("显示当前图片")
        else:
            # 显示缓存的比较结果
            cache_item = self.comparison_cache[self.cache_index]
            if cache_item["left"] is not None:
                self.display_image(self.left_canvas, cache_item["left"], self.left_markers)
            if cache_item["right"] is not None:
                self.display_image(self.right_canvas, cache_item["right"], self.right_markers)
            self.show_info(f"查看缓存结果 {self.cache_index + 1}/{len(self.comparison_cache)}")

    def toggle_zoom_mode(self):
        """切换放大器模式"""
        self.zoom_mode = not self.zoom_mode
        if self.zoom_mode:
            self.zoom_button.configure(text="放大器 (已激活)")
            self.show_info("放大器已激活，请在图像上拖动选择区域")
        else:
            self.zoom_button.configure(text="放大器")
            self.show_info("放大器已关闭")

    def start_selection(self, event, side):
        """开始选择区域"""
        if not self.zoom_mode:
            return

        # 获取正确的canvas对象
        canvas = self.left_canvas if side == "left" else self.right_canvas
        if not isinstance(canvas, tk.Canvas):
            return

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
        if not self.zoom_mode or self.selection_rect is None or self.selection_start is None:
            return

        # 获取正确的canvas对象
        canvas = self.left_canvas if side == "left" else self.right_canvas
        if not isinstance(canvas, tk.Canvas):
            return

        x1, y1 = self.selection_start
        canvas.coords(
            self.selection_rect,
            x1, y1,
            event.x, event.y
        )

    def end_selection(self, event: tk.Event, side: str) -> None:
        """结束选择并显示放大图像

        Args:
            event: 鼠标事件
            side: 图像侧（"left" 或 "right"）
        """
        if not self.zoom_mode or self.selection_rect is None or self.selection_start is None:
            return

        # 获取正确的canvas对象和图像
        canvas = self.left_canvas if side == "left" else self.right_canvas
        image = self.left_image if side == "left" else self.right_image

        if not isinstance(canvas, tk.Canvas) or image is None:
            return

        # 获取选择区域的坐标
        x1, y1 = self.selection_start
        x2, y2 = event.x, event.y

        # 确保坐标是有效的
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # 计算图像上的实际位置
        img_height, img_width = image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        x1_img = int(x1 / scale)
        y1_img = int(y1 / scale)
        x2_img = int(x2 / scale)
        y2_img = int(y2 / scale)

        # 确保坐标是有效的
        x1_img, x2_img = min(x1_img, x2_img), max(x1_img, x2_img)
        y1_img, y2_img = min(y1_img, y2_img), max(y1_img, y2_img)

        # 提取选择区域
        roi = image[y1_img:y2_img, x1_img:x2_img].copy()
        if roi.size == 0:
            return

        # 放大图像（2倍）
        new_width = roi.shape[1] * 2
        new_height = roi.shape[0] * 2
        roi = cv_resize(roi, (new_width, new_height))
        roi = cv_cvtColor(roi, COLOR_BGR2RGB)

        # 创建或更新放大窗口
        if not hasattr(self, 'zoom_window') or self.zoom_window is None:
            self.zoom_window = tk.Toplevel(self.root)
            self.zoom_window.title("放大视图")
            self.zoom_window.protocol("WM_DELETE_WINDOW", self.clear_zoom)
            self.zoom_canvas = tk.Canvas(self.zoom_window)
            self.zoom_canvas.pack()
            self.zoom_canvas.bind("<Button-1>", lambda e: self.clear_zoom())
            # 绑定ESC键
            self.zoom_window.bind("<Escape>", lambda e: self.clear_zoom())

        # 更新放大窗口位置到屏幕中心
        if self.zoom_window and self.zoom_canvas:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = roi.shape[1]
            window_height = roi.shape[0]
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            self.zoom_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

            # 显示放大的图像
            photo = ImageTk.PhotoImage(image=Image.fromarray(roi))
            self.zoom_canvas.configure(width=window_width, height=window_height)
            self.zoom_canvas.delete("all")
            self.zoom_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            setattr(self.zoom_canvas, "_photo_ref", photo)

            # 清除选择矩形
            if canvas and self.selection_rect:
                canvas.delete(self.selection_rect)
                self.selection_rect = None
                self.selection_start = None

            # 将放大窗口置于顶层
            self.zoom_window.lift()
            self.zoom_window.focus_force()

    def clear_zoom(self):
        """清除放大窗口"""
        if hasattr(self, 'zoom_window') and self.zoom_window is not None:
            self.zoom_window.destroy()
            self.zoom_window = None
            self.zoom_canvas = None

    def update_marker_coordinates(self, side, index):
        """更新标记点坐标（像素值）"""
        try:
            # 获取输入值
            if side == "left":
                if index == 0:
                    x = float(self.l1_x.get())
                    y = float(self.l1_y.get())
                else:
                    x = float(self.l2_x.get())
                    y = float(self.l2_y.get())
            else:
                if index == 0:
                    x = float(self.r1_x.get())
                    y = float(self.r1_y.get())
                else:
                    x = float(self.r2_x.get())
                    y = float(self.r2_y.get())

            # 确保坐标值在有效范围内（0-100）
            x = max(0, min(100, x))
            y = max(0, min(100, y))

            # 更新坐标
            if side == "left":
                self.left_markers[index] = (x, y)
                # 更新输入框显示
                if index == 0:
                    self.l1_x.delete(0, tk.END)
                    self.l1_x.insert(0, str(int(x)))
                    self.l1_y.delete(0, tk.END)
                    self.l1_y.insert(0, str(int(y)))
                else:
                    self.l2_x.delete(0, tk.END)
                    self.l2_x.insert(0, str(int(x)))
                    self.l2_y.delete(0, tk.END)
                    self.l2_y.insert(0, str(int(y)))
            else:
                self.right_markers[index] = (x, y)
                # 更新输入框显示
                if index == 0:
                    self.r1_x.delete(0, tk.END)
                    self.r1_x.insert(0, str(int(x)))
                    self.r1_y.delete(0, tk.END)
                    self.r1_y.insert(0, str(int(y)))
                else:
                    self.r2_x.delete(0, tk.END)
                    self.r2_x.insert(0, str(int(x)))
                    self.r2_y.delete(0, tk.END)
                    self.r2_y.insert(0, str(int(y)))

            print(f"更新坐标 - {side}图: {'L' if side == 'left' else 'R'}{index+1}({x}, {y})")

            # 更新图像显示
            if side == "left" and self.left_image is not None:
                self.display_image(self.left_canvas, self.left_image, self.left_markers)
            elif side == "right" and self.right_image is not None:
                self.display_image(self.right_canvas, self.right_image, self.right_markers)

            # 如果正在比较状态，重新比较
            if self.is_comparing:
                self.start_comparison()

        except ValueError as e:
            print(f"坐标值无效: {str(e)}")
            # 恢复原来的值
            self.update_coordinate_entries()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()