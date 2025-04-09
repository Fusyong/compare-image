import difflib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from image_processor import ImageProcessor
import os
import glob

class ImageComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像比对工具")

        # 设置窗口默认最大化
        self.root.state('zoomed')  # Windows系统使用 'zoomed'
        # 如果是在Linux/Mac系统上运行，请使用：
        # self.root.attributes('-zoomed', True)

        # 初始化图像处理器
        self.processor = ImageProcessor()
        self.processor.set_info_callback(self.show_info)  # 设置信息显示回调

        # 图像数据
        self.left_image = None
        self.right_image = None
        self.left_markers = [(5, 5), (95, 95)]  # L1, L2 (L2在右下角-5的位置)
        self.right_markers = [(5, 5), (95, 95)]  # R1, R2 (R2在右下角-5的位置)
        self.active_marker = None

        # 图片组数据
        self.left_images = []
        self.right_images = []
        self.current_index = 0

        # 放大镜参数
        self.magnifier_size = 100  # 放大区域的大小
        self.magnifier_scale = 4   # 放大倍数
        self.magnifier_visible = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # 比较状态
        self.is_comparing = False
        self.compare_mode = "compare"  # "compare" 或 "overlay"
        self.left_result = None
        self.right_result = None

        # 比较结果缓存
        self.comparison_cache = []  # 存储最近3个比较结果
        self.cache_index = -1  # 当前查看的缓存索引，-1表示不查看缓存

        self.setup_ui()
        self.bind_shortcuts()    # 绑定快捷键

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

        l2_frame = ttk.Frame(left_coords)
        l2_frame.pack(fill=tk.X)
        ttk.Label(l2_frame, text="L2: ").pack(side=tk.LEFT)
        self.l2_x = ttk.Entry(l2_frame, width=5)
        self.l2_x.pack(side=tk.LEFT)
        self.l2_y = ttk.Entry(l2_frame, width=5)
        self.l2_y.pack(side=tk.LEFT)

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

        r2_frame = ttk.Frame(right_coords)
        r2_frame.pack(fill=tk.X)
        ttk.Label(r2_frame, text="R2: ").pack(side=tk.LEFT)
        self.r2_x = ttk.Entry(r2_frame, width=5)
        self.r2_x.pack(side=tk.LEFT)
        self.r2_y = ttk.Entry(r2_frame, width=5)
        self.r2_y.pack(side=tk.LEFT)

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
        self.info_label = ttk.Label(info_frame, text="", wraplength=200)
        self.info_label.pack(fill=tk.X, padx=5, pady=2)

        # 图像显示区域（在右侧面板）
        image_frame = ttk.Frame(right_panel)
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.left_canvas = tk.Canvas(image_frame, bg='gray')
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_canvas = tk.Canvas(image_frame, bg='gray')
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 绑定事件
        self.left_canvas.bind("<Button-1>", lambda e: self.start_drag(e, "left"))
        self.left_canvas.bind("<B1-Motion>", lambda e: self.drag(e, "left"))
        self.left_canvas.bind("<ButtonRelease-1>", self.end_drag)

        self.right_canvas.bind("<Button-1>", lambda e: self.start_drag(e, "right"))
        self.right_canvas.bind("<B1-Motion>", lambda e: self.drag(e, "right"))
        self.right_canvas.bind("<ButtonRelease-1>", self.end_drag)

    def show_info(self, message):
        """显示提示信息"""
        self.info_label.config(text=message)
        self.root.update()

    def check_and_use_cache(self):
        """检查是否有可用的缓存，如果有则使用"""
        if self.left_image is None or self.right_image is None:
            return False

        # 检查是否有当前图片对的缓存
        current_cache = None
        for cache in self.comparison_cache:
            # 检查图片路径是否匹配
            if (cache.get("left_path") == self.left_images[self.current_index] and
                cache.get("right_path") == self.right_images[self.current_index] and
                cache.get("mode") == self.mode_var.get()):
                current_cache = cache
                break

        if current_cache is not None:
            # 检查图片是否被修改
            if (np.array_equal(self.left_image, current_cache.get("original_left")) and
                np.array_equal(self.right_image, current_cache.get("original_right"))):
                # 使用缓存的比较结果
                if current_cache["left"] is not None:
                    self.display_image(self.left_canvas, current_cache["left"], self.left_markers)
                if current_cache["right"] is not None:
                    self.display_image(self.right_canvas, current_cache["right"], self.right_markers)
                self.show_info("使用缓存的比较结果")
                print("使用缓存")  # 添加日志
                return True
            else:
                # 图片已修改，清除当前缓存
                self.comparison_cache = [cache for cache in self.comparison_cache if not (
                    cache.get("left_path") == self.left_images[self.current_index] and
                    cache.get("right_path") == self.right_images[self.current_index]
                )]
                print("清除缓存")  # 添加日志

        print("未找到缓存")  # 添加日志
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
                # 显示OCR进行中的提示
                self.show_info("正在OCR识别...")
                # 开始比较
                self.start_comparison()
        else:
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

    def start_comparison(self):
        """开始图像比较"""
        try:
            print("开始新的比较")  # 添加日志
            mode = self.mode_var.get()
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
                    alpha = int(self.alpha_entry.get()) / 100.0
                except ValueError:
                    alpha = 0.5  # 默认值
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

            # 缓存比较结果
            if self.left_result is not None or self.right_result is not None:
                # 添加新的当前缓存
                cache_item = {
                    "left": self.left_result.copy() if self.left_result is not None else None,
                    "right": self.right_result.copy() if self.right_result is not None else None,
                    "mode": mode,
                    "left_path": self.left_images[self.current_index],  # 保存图片路径
                    "right_path": self.right_images[self.current_index],
                    "original_left": self.left_image.copy(),  # 保存原始图片用于比较
                    "original_right": self.right_image.copy()
                }
                self.comparison_cache.append(cache_item)

                # 限制历史缓存数量
                while len(self.comparison_cache) > 3:
                    # 删除最旧的非当前缓存
                    for i, cache in enumerate(self.comparison_cache):
                        if not (
                            cache.get("left_path") == self.left_images[self.current_index] and
                            cache.get("right_path") == self.right_images[self.current_index]
                        ):
                            self.comparison_cache.pop(i)
                            break

                self.cache_index = -1  # 重置缓存索引

            # 显示结果
            if self.left_result is not None:
                self.display_image(self.left_canvas, self.left_result, self.left_markers)
            if self.right_result is not None:
                self.display_image(self.right_canvas, self.right_result, self.right_markers)

            # 根据比较结果更新提示信息
            if self.left_result is None:
                self.show_info("OCR识别失败")
            else:
                # 检查是否有差异（通过检查图像是否被修改）
                if np.array_equal(self.left_result, self.left_image if mode == "left" else self.right_image):
                    self.show_info("OCR结果相同")
                else:
                    self.show_info("已标记出差异区域并生成HTML报告")

        except Exception as e:
            self.show_info(f"比较出错: {str(e)}")
            self.is_comparing = False
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
        if 0 <= self.current_index < len(self.left_images):
            self.load_image("left", self.left_images[self.current_index])
        if 0 <= self.current_index < len(self.right_images):
            self.load_image("right", self.right_images[self.current_index])

        # 更新导航信息
        if self.left_images and self.right_images:
            self.nav_label.config(text=f"当前图片: {self.current_index + 1}/{min(len(self.left_images), len(self.right_images))}")

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

    def load_image(self, side, file_path):
        """加载图片"""
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("无法加载图片")

            if side == "left":
                self.left_image = image
                self.display_image(self.left_canvas, self.left_image, self.left_markers)
            else:
                self.right_image = image
                self.display_image(self.right_canvas, self.right_image, self.right_markers)

            self.update_coordinate_entries()
            self.show_info(f"已加载{side}图: {os.path.basename(file_path)}")
        except Exception as e:
            self.show_info(f"加载图片失败: {str(e)}")

    def display_image(self, canvas, image, markers):
        # 调整图像大小以适应画布
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300

        image = cv2.resize(image, (canvas_width, canvas_height))  # pylint: disable=no-member
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

        # 绘制标记点
        for i, (x, y) in enumerate(markers):
            x = int(x * canvas_width / 100)
            y = int(y * canvas_height / 100)
            cv2.drawMarker(image, (x, y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)  # pylint: disable=no-member
            cv2.putText(image, # pylint: disable=no-member
                        f"{'L' if canvas == self.left_canvas else 'R'}{i+1}",
                       (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, # pylint: disable=no-member
                       0.5, (255, 0, 0), 1)

        # 显示图像
        photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # 保持引用

    def start_drag(self, event, side):
        canvas = self.left_canvas if side == "left" else self.right_canvas
        markers = self.left_markers if side == "left" else self.right_markers
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
                self.update_magnifier(event, canvas, image)
                break

    def drag(self, event, side):
        if not self.active_marker or self.active_marker[0] != side:
            return

        canvas = self.left_canvas if side == "left" else self.right_canvas
        markers = self.left_markers if side == "left" else self.right_markers
        image = self.left_image if side == "left" else self.right_image

        # 更新标记点位置
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        x_rel = min(max(event.x / canvas_width * 100, 0), 100)
        y_rel = min(max(event.y / canvas_height * 100, 0), 100)

        markers[self.active_marker[1]] = (x_rel, y_rel)

        # 更新显示
        if side == "left" and self.left_image is not None:
            self.display_image(self.left_canvas, self.left_image, self.left_markers)
        elif side == "right" and self.right_image is not None:
            self.display_image(self.right_canvas, self.right_image, self.right_markers)

        # 更新放大镜
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
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

    def update_coordinate_entries(self):
        # 更新左图坐标
        self.l1_x.delete(0, tk.END)
        self.l1_x.insert(0, f"{self.left_markers[0][0]:.1f}")
        self.l1_y.delete(0, tk.END)
        self.l1_y.insert(0, f"{self.left_markers[0][1]:.1f}")

        self.l2_x.delete(0, tk.END)
        self.l2_x.insert(0, f"{self.left_markers[1][0]:.1f}")
        self.l2_y.delete(0, tk.END)
        self.l2_y.insert(0, f"{self.left_markers[1][1]:.1f}")

        # 更新右图坐标
        self.r1_x.delete(0, tk.END)
        self.r1_x.insert(0, f"{self.right_markers[0][0]:.1f}")
        self.r1_y.delete(0, tk.END)
        self.r1_y.insert(0, f"{self.right_markers[0][1]:.1f}")

        self.r2_x.delete(0, tk.END)
        self.r2_x.insert(0, f"{self.right_markers[1][0]:.1f}")
        self.r2_y.delete(0, tk.END)
        self.r2_y.insert(0, f"{self.right_markers[1][1]:.1f}")

    def update_magnifier(self, event, canvas, image):
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
        roi = cv2.resize(roi, (self.magnifier_size, self.magnifier_size))  # pylint: disable=no-member
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

        # 在放大的图像中心绘制十字线
        center = self.magnifier_size // 2
        cv2.line(roi, (center, 0), (center, self.magnifier_size), (255, 0, 0), 1)  # pylint: disable=no-member
        cv2.line(roi, (0, center), (self.magnifier_size, center), (255, 0, 0), 1)  # pylint: disable=no-member

        # 创建或更新放大镜窗口
        if not hasattr(self, 'magnifier_window'):
            self.magnifier_window = tk.Toplevel(self.root)
            self.magnifier_window.overrideredirect(True)
            self.magnifier_canvas = tk.Canvas(self.magnifier_window,
                                           width=self.magnifier_size,
                                           height=self.magnifier_size)
            self.magnifier_canvas.pack()

        # 更新放大镜位置
        screen_x = self.root.winfo_rootx() + canvas.winfo_x() + event.x
        screen_y = self.root.winfo_rooty() + canvas.winfo_y() + event.y
        self.magnifier_window.geometry(f"{self.magnifier_size}x{self.magnifier_size}+{screen_x+20}+{screen_y+20}")

        # 显示放大的图像
        magnifier_image = ImageTk.PhotoImage(image=Image.fromarray(roi))
        self.magnifier_canvas.create_image(0, 0, anchor=tk.NW, image=magnifier_image)
        self.magnifier_canvas.image = magnifier_image  # type: ignore

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

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()