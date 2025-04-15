"""
图像比对工具
"""
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, ImageTk
import numpy as np

from image_processor import ImageProcessor
from cv_utils import (
    ImageArray, COLOR_BGR2RGB, MARKER_CROSS, FONT_HERSHEY_SIMPLEX,
    cv_imread, cv_resize, cv_cvtColor, cv_drawMarker, cv_putText
)

# 类型别名定义
Coordinates = Tuple[float, float]  # 坐标点 (x, y)
MarkerList = List[Coordinates]  # 标记点列表
CacheItem = Dict[str, Any]  # 缓存项类型

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
        self.left_markers: MarkerList = []  # L1, L2
        self.right_markers: MarkerList = []  # R1, R2
        self.active_marker: Optional[Tuple[str, int]] = None  # (side, index)
        self.left_result: Optional[ImageArray] = None
        self.right_result: Optional[ImageArray] = None

        # 图片组数据
        self.left_images: List[str] = []
        self.right_images: List[str] = []
        self.current_index: int = 0

        # 标记点放大器参数
        self.marker_magnifier_size: int = 100  # 标记点放大区域的大小
        self.marker_magnifier_scale: int = 4   # 标记点放大倍数
        self.marker_magnifier_visible: bool = False
        self.marker_magnifier_window: Optional[tk.Toplevel] = None
        self.marker_magnifier_canvas: Optional[tk.Canvas] = None
        self.last_mouse_x: int = 0
        self.last_mouse_y: int = 0

        # 区域放大器参数
        self.area_magnifier_window: Optional[tk.Toplevel] = None
        self.area_magnifier_canvas: Optional[tk.Canvas] = None
        self.selection_start: Optional[Tuple[int, int]] = None
        self.selection_end = None
        self.selection_rect: Optional[int] = None

        # 比较状态
        self.is_comparing: bool = False
        self.compare_mode: str = "compare"  # "compare" 或 "overlay" 或 "ocr"

        # 比较结果缓存
        self.comparison_cache: List[CacheItem] = []
        self.cache_index: int = -1

        # 初始化UI
        self.setup_ui()
        self.bind_shortcuts()

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

        # 添加页码输入框
        page_input_frame = ttk.Frame(toolbar)
        page_input_frame.pack(fill=tk.X, padx=5, pady=2)

        # 左图页码
        left_page_frame = ttk.Frame(page_input_frame)
        left_page_frame.pack(side=tk.LEFT, expand=True)
        ttk.Label(left_page_frame, text="左页码:").pack(side=tk.LEFT)
        self.left_page_entry = ttk.Entry(left_page_frame, width=5)
        self.left_page_entry.pack(side=tk.LEFT, padx=2)
        self.left_page_entry.insert(0, "1")
        self.left_page_entry.bind("<Return>", lambda e: self.jump_to_page("left"))

        # 右图页码
        right_page_frame = ttk.Frame(page_input_frame)
        right_page_frame.pack(side=tk.LEFT, expand=True)
        ttk.Label(right_page_frame, text="右页码:").pack(side=tk.LEFT)
        self.right_page_entry = ttk.Entry(right_page_frame, width=5)
        self.right_page_entry.pack(side=tk.LEFT, padx=2)
        self.right_page_entry.insert(0, "1")
        self.right_page_entry.bind("<Return>", lambda e: self.jump_to_page("right"))

        # 坐标输入区域 - 移动到这里，放在按钮下方
        coords_frame = ttk.LabelFrame(toolbar, text="校准点坐标")
        coords_frame.pack(fill=tk.X, padx=5, pady=2)

        # 左图坐标
        left_coords = ttk.Frame(coords_frame)
        left_coords.pack(fill=tk.X, padx=5, pady=2)

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
        # 绑定焦点事件
        self.l1_x.bind("<FocusIn>", lambda e: self.show_marker_magnifier("left", 0))
        self.l1_y.bind("<FocusIn>", lambda e: self.show_marker_magnifier("left", 0))
        self.l1_x.bind("<FocusOut>", self.hide_marker_magnifier)
        self.l1_y.bind("<FocusOut>", self.hide_marker_magnifier)

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
        # 绑定焦点事件
        self.l2_x.bind("<FocusIn>", lambda e: self.show_marker_magnifier("left", 1))
        self.l2_y.bind("<FocusIn>", lambda e: self.show_marker_magnifier("left", 1))
        self.l2_x.bind("<FocusOut>", self.hide_marker_magnifier)
        self.l2_y.bind("<FocusOut>", self.hide_marker_magnifier)

        # 右图坐标
        right_coords = ttk.Frame(coords_frame)
        right_coords.pack(fill=tk.X, padx=5, pady=2)

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
        # 绑定焦点事件
        self.r1_x.bind("<FocusIn>", lambda e: self.show_marker_magnifier("right", 0))
        self.r1_y.bind("<FocusIn>", lambda e: self.show_marker_magnifier("right", 0))
        self.r1_x.bind("<FocusOut>", self.hide_marker_magnifier)
        self.r1_y.bind("<FocusOut>", self.hide_marker_magnifier)

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
        # 绑定焦点事件
        self.r2_x.bind("<FocusIn>", lambda e: self.show_marker_magnifier("right", 1))
        self.r2_y.bind("<FocusIn>", lambda e: self.show_marker_magnifier("right", 1))
        self.r2_x.bind("<FocusOut>", self.hide_marker_magnifier)
        self.r2_y.bind("<FocusOut>", self.hide_marker_magnifier)

        # 添加模式选择 - 移到坐标输入区域后面
        mode_frame = ttk.LabelFrame(toolbar, text="比较模式")
        mode_frame.pack(fill=tk.X, padx=5, pady=2)

        # 模式选择 - 第一行
        mode_select_frame = ttk.Frame(mode_frame)
        mode_select_frame.pack(fill=tk.X, padx=5, pady=2)
        self.mode_var = tk.StringVar(value="overlay")
        ttk.Radiobutton(mode_select_frame, text="叠加", variable=self.mode_var, value="overlay").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_select_frame, text="像素", variable=self.mode_var, value="compare").pack(side=tk.LEFT, padx=5)
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
        self.compare_button = ttk.Button(toolbar, text="比较/原图 Alt+Q", command=self.toggle_compare)
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
        self.zoom_button = ttk.Button(toolbar, text="放大器 Alt+Z", command=self.toggle_zoom_mode)
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

        return False

    def toggle_compare(self):
        """切换比较状态"""
        if self.left_image is None or self.right_image is None:
            self.show_info("请先加载左右两张图片")
            return

        if not self.is_comparing:
            self.is_comparing = True
            self.compare_button.configure(text="原图 Alt+Q")

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
            self.compare_button.configure(text="比较 Alt+Q")
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
            mode = self.mode_var.get()

            # 检查图像和坐标的有效性
            if self.left_image is None or self.right_image is None:
                print("错误：图像未加载")
                self.show_info("请先加载两张图片")
                return

            if not self.left_markers or not self.right_markers:
                print("错误：标记点无效")
                self.show_info("请先设置有效的标记点")
                return

            # 执行比较
            if mode == "compare":
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
            if self.left_result is None or self.right_result is None:
                self.show_info("比较失败，请检查图像和标记点")
                return

            # 缓存比较结果
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
                self.compare_button.config(text="比较 Alt+Q")

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
        # 根据页码输入框的值加载图片
        try:
            left_page = int(self.left_page_entry.get()) - 1
            if 0 <= left_page < len(self.left_images):
                self.load_image("left", self.left_images[left_page])
        except ValueError:
            pass

        try:
            right_page = int(self.right_page_entry.get()) - 1
            if 0 <= right_page < len(self.right_images):
                self.load_image("right", self.right_images[right_page])
        except ValueError:
            pass

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
                # 如果是第一次加载图片，初始化标记点
                if not self.left_markers:
                    self.left_markers = [(50, 50), (image.shape[1] - 50, image.shape[0] - 50)]
                if self.left_canvas:
                    self.display_image(self.left_canvas, image, self.left_markers)
                    self.left_canvas.update_idletasks()
            else:
                self.right_image = image
                # 如果是第一次加载图片，初始化标记点
                if not self.right_markers:
                    self.right_markers = [(50, 50), (image.shape[1] - 50, image.shape[0] - 50)]
                if self.right_canvas:
                    self.display_image(self.right_canvas, image, self.right_markers)
                    self.right_canvas.update_idletasks()

            # 更新坐标输入框显示
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
        if image is None:
            return

        # 获取画布尺寸
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
            canvas.configure(width=canvas_width, height=canvas_height)

        # 获取两张图片中较大的尺寸
        max_img_width = max(
            self.left_image.shape[1] if self.left_image is not None else 0,
            self.right_image.shape[1] if self.right_image is not None else 0
        )
        max_img_height = max(
            self.left_image.shape[0] if self.left_image is not None else 0,
            self.right_image.shape[0] if self.right_image is not None else 0
        )

        # 计算两个画布的总宽度（减去边距）
        total_canvas_width = self.left_canvas.winfo_width() + self.right_canvas.winfo_width()

        # 使用较大图片的尺寸和总画布宽度计算统一的缩放比例
        if max_img_width > 0 and max_img_height > 0:
            # 计算水平和垂直方向的缩放比例
            scale_w = total_canvas_width / (2 * max_img_width)  # 除以2因为有两个画布
            scale_h = canvas_height / max_img_height
            # 使用较小的缩放比例以确保图像完全显示
            scale = min(scale_w, scale_h)
        else:
            scale = 1.0

        # 计算当前图片缩放后的尺寸
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)

        # 计算偏移量，使图像在画布中居中
        if canvas == self.left_canvas:
            offset_x = canvas_width - new_width  # 靠右对齐
            offset_y = (canvas_height - new_height) // 2  # 垂直居中
        else:
            offset_x = 0  # 靠左对齐
            offset_y = (canvas_height - new_height) // 2  # 垂直居中

        # 清除所有图像相关的内容，但保留scale_info
        for item in canvas.find_all():
            if "scale_info" not in canvas.gettags(item):
                canvas.delete(item)

        # 更新缩放和偏移信息
        scale_items = canvas.find_withtag("scale_info")
        if scale_items:
            canvas.itemconfig(scale_items[0], text=f"{scale},{offset_x},{offset_y},{new_width},{new_height}")
        else:
            canvas.create_text(0, 0, text=f"{scale},{offset_x},{offset_y},{new_width},{new_height}",
                             tags="scale_info", state="hidden")

        # 调整图像大小
        resized_image = cv_resize(image, (new_width, new_height))
        rgb_image = cv_cvtColor(resized_image, COLOR_BGR2RGB)

        # 绘制标记点
        for i, (x, y) in enumerate(markers):
            # 将像素坐标转换为画布坐标
            px = int(x * scale) + offset_x
            py = int(y * scale) + offset_y

            # 确保坐标在画布范围内
            px = max(offset_x, min(offset_x + new_width - 1, px))
            py = max(offset_y, min(offset_y + new_height - 1, py))

            cv_drawMarker(rgb_image, (int((px - offset_x)), int((py - offset_y))), (255, 0, 0), MARKER_CROSS, 20, 2)
            cv_putText(rgb_image,
                     f"{'L' if canvas == self.left_canvas else 'R'}{i+1}",
                     (int((px - offset_x))+5, int((py - offset_y))-5),
                     FONT_HERSHEY_SIMPLEX,
                     0.5, (255, 0, 0), 1)

        # 显示图像
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_image))
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=photo, tags="image")

        # 使用setattr来避免类型检查错误
        setattr(canvas, "_photo_ref", photo)

    def _get_scale_info(self, canvas: tk.Canvas) -> Optional[Dict[str, float]]:
        """从canvas的tag中获取缩放信息

        Args:
            canvas: 目标画布

        Returns:
            包含缩放信息的字典，如果没有找到则返回None
        """
        scale_items = canvas.find_withtag("scale_info")
        if not scale_items:
            return None

        scale_text = canvas.itemcget(scale_items[0], "text")
        if not scale_text:
            return None

        try:
            scale, offset_x, offset_y, new_width, new_height = map(float, scale_text.split(","))
            return {
                'scale': scale,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'new_width': new_width,
                'new_height': new_height
            }
        except (ValueError, IndexError):
            return None

    def on_left_click(self, event):
        """处理左图点击事件"""
        if event.state & 0x4:  # Ctrl键被按下
            self.start_selection(event, "left")
        elif self.zoom_mode:
            self.start_selection(event, "left")
        else:
            self.start_drag(event, "left")

    def on_left_drag(self, event):
        """处理左图拖动事件"""
        if event.state & 0x4:  # Ctrl键被按下
            self.update_selection(event, "left")
        elif self.zoom_mode:
            self.update_selection(event, "left")
        else:
            self.drag(event, "left")

    def on_left_release(self, event):
        """处理左图释放事件"""
        if event.state & 0x4:  # Ctrl键被按下
            self.end_selection(event, "left")
        elif self.zoom_mode:
            self.end_selection(event, "left")
        else:
            self.end_drag(event)

    def on_right_click(self, event):
        """处理右图点击事件"""
        if event.state & 0x4:  # Ctrl键被按下
            self.start_selection(event, "right")
        elif self.zoom_mode:
            self.start_selection(event, "right")
        else:
            self.start_drag(event, "right")

    def on_right_drag(self, event):
        """处理右图拖动事件"""
        if event.state & 0x4:  # Ctrl键被按下
            self.update_selection(event, "right")
        elif self.zoom_mode:
            self.update_selection(event, "right")
        else:
            self.drag(event, "right")

    def on_right_release(self, event):
        """处理右图释放事件"""
        if event.state & 0x4:  # Ctrl键被按下
            self.end_selection(event, "right")
        elif self.zoom_mode:
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

        if image is None:
            return

        # 获取画布的缩放信息
        scale_info = self._get_scale_info(canvas)
        if not scale_info:
            return

        scale = scale_info['scale']
        offset_x = scale_info['offset_x']
        offset_y = scale_info['offset_y']

        # 检查是否点击了标记点（增加点击范围到15像素）
        for i, (mx, my) in enumerate(markers):
            # 将标记点坐标转换为画布坐标
            px = int(mx * scale) + offset_x
            py = int(my * scale) + offset_y

            # 检查点击位置是否在标记点附近
            if abs(event.x - px) < 15 and abs(event.y - py) < 15:
                self.active_marker = (side, i)
                self.marker_magnifier_visible = True
                self.last_mouse_x = event.x
                self.last_mouse_y = event.y
                if image is not None:
                    self.update_marker_magnifier(event, canvas, image)
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

        if image is None:
            return

        # 获取画布的缩放信息
        scale_info = self._get_scale_info(canvas)
        if not scale_info:
            return

        scale = scale_info['scale']
        offset_x = scale_info['offset_x']
        offset_y = scale_info['offset_y']

        # 将画布坐标转换为图像坐标
        x = int((event.x - offset_x) / scale)
        y = int((event.y - offset_y) / scale)

        # 确保坐标在图像范围内
        x = max(0, min(image.shape[1] - 1, x))
        y = max(0, min(image.shape[0] - 1, y))

        # 更新标记点位置
        markers[self.active_marker[1]] = (x, y)

        # 更新显示
        if side == "left" and self.left_image is not None:
            self.display_image(self.left_canvas, image, self.left_markers)
        elif side == "right" and self.right_image is not None:
            self.display_image(self.right_canvas, image, self.right_markers)

        # 更新放大镜
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        if image is not None:
            self.update_marker_magnifier(event, canvas, image)

        # 更新坐标输入框
        self.update_coordinate_entries()

    def end_drag(self, event):
        """结束拖动标记点"""
        self.active_marker = None
        self.marker_magnifier_visible = False
        # 清除标记点放大器
        if hasattr(self, 'marker_magnifier_window') and self.marker_magnifier_window is not None:
            self.marker_magnifier_window.destroy()
            self.marker_magnifier_window = None
            self.marker_magnifier_canvas = None

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

    def update_marker_magnifier(self, event: tk.Event, canvas: tk.Canvas, image: ImageArray) -> None:
        """更新标记点放大器显示

        Args:
            event: 鼠标事件
            canvas: 目标画布
            image: 要显示的图像（原图或比较结果）
        """
        if not self.marker_magnifier_visible or image is None or not self.active_marker:
            return

        # 获取画布的缩放信息
        scale_info = self._get_scale_info(canvas)
        if not scale_info:
            return

        scale = scale_info['scale']
        offset_x = int(scale_info['offset_x'])
        offset_y = int(scale_info['offset_y'])

        # 获取当前活动的标记点坐标
        markers = self.left_markers if canvas == self.left_canvas else self.right_markers
        marker_x, marker_y = markers[self.active_marker[1]]

        # 计算放大区域的范围（以标记点为中心）
        half_size = self.marker_magnifier_size // (2 * self.marker_magnifier_scale)
        img_x = int(marker_x - half_size)
        img_y = int(marker_y - half_size)

        # 确保范围在图像内
        img_x = max(0, min(img_x, image.shape[1] - 2 * half_size))
        img_y = max(0, min(img_y, image.shape[0] - 2 * half_size))

        # 提取放大区域
        roi = image[img_y:img_y + 2 * half_size, img_x:img_x + 2 * half_size].copy()
        if roi.size == 0:
            return

        # 放大图像
        magnified_size = (self.marker_magnifier_size, self.marker_magnifier_size)
        roi = cv_resize(roi, magnified_size)
        roi = cv_cvtColor(roi, COLOR_BGR2RGB)

        # 在放大的图像中心绘制十字线
        center = self.marker_magnifier_size // 2
        cv_drawMarker(roi, (center, center), (255, 0, 0), MARKER_CROSS, 30, 2)

        # 创建或更新标记点放大器窗口
        if not hasattr(self, 'marker_magnifier_window') or not self.marker_magnifier_window:
            self.marker_magnifier_window = tk.Toplevel(self.root)
            self.marker_magnifier_window.overrideredirect(True)
            self.marker_magnifier_canvas = tk.Canvas(self.marker_magnifier_window,
                                           width=self.marker_magnifier_size,
                                           height=self.marker_magnifier_size)
            self.marker_magnifier_canvas.pack()

        # 将标记点坐标转换为画布坐标
        canvas_x = int(marker_x * scale) + offset_x
        canvas_y = int(marker_y * scale) + offset_y

        # 计算放大器窗口在屏幕上的位置（确保使用整数坐标）
        root_x = int(self.root.winfo_rootx())
        root_y = int(self.root.winfo_rooty())
        canvas_root_x = int(canvas.winfo_x())
        canvas_root_y = int(canvas.winfo_y())

        screen_x = root_x + canvas_root_x + canvas_x
        screen_y = root_y + canvas_root_y + canvas_y

        if self.marker_magnifier_window:
            # 调整放大器位置，使其不遮挡标记点
            if canvas == self.left_canvas:
                # 左侧画布：放大器显示在右侧
                magnifier_x = screen_x + 20
            else:
                # 右侧画布：放大器显示在左侧
                magnifier_x = screen_x - 20

            magnifier_y = screen_y - self.marker_magnifier_size // 2

            # 确保所有坐标都是整数
            geometry_str = f"{self.marker_magnifier_size}x{self.marker_magnifier_size}+{int(magnifier_x)}+{int(magnifier_y)}"
            self.marker_magnifier_window.geometry(geometry_str)

            # 显示放大的图像
            if self.marker_magnifier_canvas:
                photo = ImageTk.PhotoImage(image=Image.fromarray(roi))
                self.marker_magnifier_canvas.delete("all")
                self.marker_magnifier_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                setattr(self.marker_magnifier_canvas, "_photo_ref", photo)

    def hide_marker_magnifier(self, event=None) -> None:
        """隐藏标记点放大器"""
        self.active_marker = None
        self.marker_magnifier_visible = False
        # 清除标记点放大器
        if hasattr(self, 'marker_magnifier_window') and self.marker_magnifier_window is not None:
            self.marker_magnifier_window.destroy()
            self.marker_magnifier_window = None
            self.marker_magnifier_canvas = None

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
            self.zoom_button.configure(text="放大器 Alt+Z (已激活)")
            self.show_info("放大器已激活，请在图像上拖动选择区域")
        else:
            self.zoom_button.configure(text="放大器 Alt+Z")
            self.show_info("放大器已关闭")

    def start_selection(self, event, side):
        """开始选择区域"""
        # 获取正确的canvas对象
        canvas = self.left_canvas if side == "left" else self.right_canvas
        if not isinstance(canvas, tk.Canvas):
            return

        self.selection_start = (event.x, event.y)
        self.selection_rect = canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline='red', width=2
        )

        # 显示提示信息
        if event.state & 0x4:  # Ctrl键被按下
            self.show_info("按住Ctrl键拖动选择区域进行放大")

    def update_selection(self, event: tk.Event, side: str) -> None:
        """更新选择区域

        Args:
            event: 鼠标事件
            side: 图像侧（"left" 或 "right"）
        """
        # 获取正确的canvas对象
        canvas = self.left_canvas if side == "left" else self.right_canvas
        if not isinstance(canvas, tk.Canvas) or self.selection_rect is None or self.selection_start is None:
            return

        x1, y1 = self.selection_start
        canvas.coords(
            self.selection_rect,
            x1, y1,
            event.x, event.y
        )

    def end_selection(self, event: tk.Event, side: str) -> None:
        """结束选择并显示区域放大器

        Args:
            event: 鼠标事件
            side: 图像侧（"left" 或 "right"）
        """
        # 获取正确的canvas对象和图像
        canvas = self.left_canvas if side == "left" else self.right_canvas

        # 根据当前状态选择正确的图像
        if self.is_comparing:
            if self.mode_var.get() == "ocr":
                # OCR模式下显示原图
                image = self.left_image if side == "left" else self.right_image
            else:
                # 其他比较模式下显示比较结果
                image = self.left_result if side == "left" else self.right_result
        else:
            # 非比较状态下显示原图
            image = self.left_image if side == "left" else self.right_image

        if not isinstance(canvas, tk.Canvas) or image is None or self.selection_rect is None or self.selection_start is None:
            return

        # 获取选择区域的坐标
        x1, y1 = self.selection_start
        x2, y2 = event.x, event.y

        # 获取画布的缩放信息
        scale_info = self._get_scale_info(canvas)
        if not scale_info:
            return

        scale = scale_info['scale']
        offset_x = int(scale_info['offset_x'])
        offset_y = int(scale_info['offset_y'])

        # 计算图像上的实际位置（考虑画布偏移）
        x1_img = int((x1 - offset_x) / scale)
        y1_img = int((y1 - offset_y) / scale)
        x2_img = int((x2 - offset_x) / scale)
        y2_img = int((y2 - offset_y) / scale)

        # 确保坐标是有效的
        x1_img, x2_img = min(x1_img, x2_img), max(x1_img, x2_img)
        y1_img, y2_img = min(y1_img, y2_img), max(y1_img, y2_img)

        # 确保坐标在图像范围内
        img_height, img_width = image.shape[:2]
        x1_img = max(0, min(x1_img, img_width - 1))
        y1_img = max(0, min(y1_img, img_height - 1))
        x2_img = max(0, min(x2_img, img_width - 1))
        y2_img = max(0, min(y2_img, img_height - 1))

        # 提取选择区域
        roi = image[y1_img:y2_img, x1_img:x2_img].copy()
        if roi.size == 0:
            return

        # 放大图像（2倍）
        new_width = roi.shape[1] * 2
        new_height = roi.shape[0] * 2
        roi = cv_resize(roi, (new_width, new_height))
        roi = cv_cvtColor(roi, COLOR_BGR2RGB)

        # 创建或更新区域放大器窗口
        if not hasattr(self, 'area_magnifier_window') or self.area_magnifier_window is None:
            self.area_magnifier_window = tk.Toplevel(self.root)
            self.area_magnifier_window.title("区域放大器")
            self.area_magnifier_window.protocol("WM_DELETE_WINDOW", self.clear_area_magnifier)
            self.area_magnifier_canvas = tk.Canvas(self.area_magnifier_window)
            self.area_magnifier_canvas.pack()
            self.area_magnifier_canvas.bind("<Button-1>", lambda e: self.clear_area_magnifier())
            # 绑定ESC键
            self.area_magnifier_window.bind("<Escape>", lambda e: self.clear_area_magnifier())

        # 更新区域放大器窗口位置到屏幕中心
        if self.area_magnifier_window and self.area_magnifier_canvas:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = roi.shape[1]
            window_height = roi.shape[0]
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            self.area_magnifier_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

            # 显示放大的图像
            photo = ImageTk.PhotoImage(image=Image.fromarray(roi))
            self.area_magnifier_canvas.configure(width=window_width, height=window_height)
            self.area_magnifier_canvas.delete("all")
            self.area_magnifier_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            setattr(self.area_magnifier_canvas, "_photo_ref", photo)

            # 清除选择矩形
            if canvas and self.selection_rect:
                canvas.delete(self.selection_rect)
                self.selection_rect = None
                self.selection_start = None

            # 将区域放大器窗口置于顶层
            self.area_magnifier_window.lift()
            self.area_magnifier_window.focus_force()

    def clear_area_magnifier(self):
        """清除区域放大器窗口"""
        if hasattr(self, 'area_magnifier_window') and self.area_magnifier_window is not None:
            self.area_magnifier_window.destroy()
            self.area_magnifier_window = None
            self.area_magnifier_canvas = None

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

            # 获取图像尺寸
            image = self.left_image if side == "left" else self.right_image
            if image is None:
                return
            img_height, img_width = image.shape[:2]

            # 确保坐标值在图像范围内
            x = max(0, min(img_width - 1, x))
            y = max(0, min(img_height - 1, y))

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

            # 更新图像显示
            if side == "left" and self.left_image is not None:
                self.display_image(self.left_canvas, self.left_image, self.left_markers)
            elif side == "right" and self.right_image is not None:
                self.display_image(self.right_canvas, self.right_image, self.right_markers)

            # 如果正在比较状态，重新比较
            if self.is_comparing:
                self.start_comparison()

            # 更新放大镜位置
            self.show_marker_magnifier(side, index)

        except ValueError as e:
            print(f"坐标值无效: {str(e)}")
            # 恢复原来的值
            self.update_coordinate_entries()

    def jump_to_page(self, side: str) -> None:
        """跳转到指定页码

        Args:
            side: 图像侧（"left" 或 "right"）
        """
        try:
            if side == "left":
                page = int(self.left_page_entry.get()) - 1
                max_page = len(self.left_images)
            else:
                page = int(self.right_page_entry.get()) - 1
                max_page = len(self.right_images)

            if 0 <= page < max_page:
                # 更新当前索引
                self.current_index = page

                # 只加载输入侧的图像
                if side == "left" and 0 <= page < len(self.left_images):
                    self.load_image("left", self.left_images[page])
                elif side == "right" and 0 <= page < len(self.right_images):
                    self.load_image("right", self.right_images[page])

                # 只更新输入侧的页码显示
                if side == "left":
                    self.left_page_entry.delete(0, tk.END)
                    self.left_page_entry.insert(0, str(page + 1))
                else:
                    self.right_page_entry.delete(0, tk.END)
                    self.right_page_entry.insert(0, str(page + 1))
                self.show_info(f"已跳转到第{page + 1}页")
            else:
                self.show_info(f"页码必须在1-{max_page}之间")
                # 恢复原来的页码
                if side == "left":
                    self.left_page_entry.delete(0, tk.END)
                    self.left_page_entry.insert(0, str(self.current_index + 1))
                else:
                    self.right_page_entry.delete(0, tk.END)
                    self.right_page_entry.insert(0, str(self.current_index + 1))
        except ValueError:
            self.show_info("请输入有效的页码")
            # 恢复原来的页码
            if side == "left":
                self.left_page_entry.delete(0, tk.END)
                self.left_page_entry.insert(0, str(self.current_index + 1))
            else:
                self.right_page_entry.delete(0, tk.END)
                self.right_page_entry.insert(0, str(self.current_index + 1))

    def update_page_entries(self) -> None:
        """更新页码输入框的值"""
        if self.left_page_entry and self.right_page_entry:
            # 保持当前页码不变，只更新显示
            self.left_page_entry.delete(0, tk.END)
            self.right_page_entry.delete(0, tk.END)
            self.left_page_entry.insert(0, str(self.current_index + 1))
            self.right_page_entry.insert(0, str(self.current_index + 1))

    def navigate_images(self, direction):
        """导航到上一张或下一张图片"""
        if not self.left_images or not self.right_images:
            return

        try:
            # 获取当前页码
            left_page = int(self.left_page_entry.get()) - 1
            right_page = int(self.right_page_entry.get()) - 1

            # 计算新页码
            new_left_page = left_page + direction
            new_right_page = right_page + direction

            # 检查新页码是否有效
            if 0 <= new_left_page < len(self.left_images) and 0 <= new_right_page < len(self.right_images):
                # 更新页码输入框
                self.left_page_entry.delete(0, tk.END)
                self.left_page_entry.insert(0, str(new_left_page + 1))
                self.right_page_entry.delete(0, tk.END)
                self.right_page_entry.insert(0, str(new_right_page + 1))

                # 加载新图片
                self.load_image("left", self.left_images[new_left_page])
                self.load_image("right", self.right_images[new_right_page])

                # 更新当前索引（用于导航信息显示）
                self.current_index = new_left_page

                # 如果当前处于比较状态，开始比较
                if self.is_comparing:
                    # 先检查是否有可用的缓存
                    if not self.check_and_use_cache():
                        self.start_comparison()
            else:
                self.show_info("已到达图片边界")
        except ValueError:
            self.show_info("页码值无效")

    def bind_shortcuts(self):
        """绑定快捷键"""
        self.root.bind("<Prior>", lambda e: self.navigate_images(-1))  # PgUp
        self.root.bind("<Next>", lambda e: self.navigate_images(1))    # PgDn
        self.root.bind("<Alt-q>", lambda e: self.toggle_compare())  # Alt+Q
        self.root.bind("<Alt-z>", lambda e: self.toggle_zoom_mode())  # Alt+Z

    def show_marker_magnifier(self, side: str, index: int) -> None:
        """显示标记点放大器

        Args:
            side: 图像侧（"left" 或 "right"）
            index: 标记点索引（0 或 1）
        """
        canvas = self.left_canvas if side == "left" else self.right_canvas
        image = self.left_image if side == "left" else self.right_image
        markers = self.left_markers if side == "left" else self.right_markers

        if image is None or not markers:
            return

        # 设置活动标记点
        self.active_marker = (side, index)
        self.marker_magnifier_visible = True

        # 创建一个虚拟事件来更新放大镜
        class DummyEvent(tk.Event):
            def __init__(self, x, y):
                super().__init__()
                self.x = x
                self.y = y

        # 获取标记点坐标
        marker_x, marker_y = markers[index]

        # 获取画布的缩放信息
        scale_info = self._get_scale_info(canvas)
        if not scale_info:
            return

        scale = scale_info['scale']
        offset_x = int(scale_info['offset_x'])
        offset_y = int(scale_info['offset_y'])

        # 将标记点坐标转换为画布坐标
        canvas_x = int(marker_x * scale) + offset_x
        canvas_y = int(marker_y * scale) + offset_y

        # 创建虚拟事件
        event = DummyEvent(canvas_x, canvas_y)
        self.last_mouse_x = canvas_x
        self.last_mouse_y = canvas_y

        # 更新放大镜
        self.update_marker_magnifier(event, canvas, image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()