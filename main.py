import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from image_processor import ImageProcessor

class ImageComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像比对工具")

        # 初始化图像处理器
        self.processor = ImageProcessor()
        self.processor.set_info_callback(self.show_info)  # 设置信息显示回调

        # 图像数据
        self.left_image = None
        self.right_image = None
        self.left_markers = [(5, 5), (95, 95)]  # L1, L2 (L2在右下角-5的位置)
        self.right_markers = [(5, 5), (95, 95)]  # R1, R2 (R2在右下角-5的位置)
        self.active_marker = None

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

        self.setup_ui()

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
        ttk.Button(buttons_frame, text="加载左图", command=lambda: self.load_image("left")).pack(side=tk.LEFT, expand=True, padx=(0,2))
        ttk.Button(buttons_frame, text="加载右图", command=lambda: self.load_image("right")).pack(side=tk.LEFT, expand=True, padx=(2,0))

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
        mode_frame = ttk.Frame(toolbar)
        mode_frame.pack(fill=tk.X, padx=5, pady=2)
        self.mode_var = tk.StringVar(value="compare")
        ttk.Radiobutton(mode_frame, text="像素比较", variable=self.mode_var, value="compare").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="叠加模式", variable=self.mode_var, value="overlay").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="OCR比较", variable=self.mode_var, value="ocr").pack(side=tk.LEFT)

        # 叠加模式的透明度控制
        alpha_frame = ttk.Frame(toolbar)
        alpha_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(alpha_frame, text="叠加透明度:").pack(side=tk.LEFT)
        self.alpha_scale = ttk.Scale(alpha_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.alpha_scale.set(50)  # 默认透明度0.5
        self.alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.compare_button = ttk.Button(toolbar, text="开始比较", command=self.toggle_compare)
        self.compare_button.pack(fill=tk.X, padx=5, pady=2)

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

    def toggle_compare(self):
        """切换比较状态"""
        if self.left_image is None or self.right_image is None:
            self.show_info("请先加载左右两张图片")
            return

        if not self.is_comparing:
            self.is_comparing = True
            self.compare_button.configure(text="停止比较")

            # 显示OCR进行中的提示
            self.show_info("正在OCR识别...")

            # 开始比较
            self.start_comparison()
        else:
            self.is_comparing = False
            self.compare_button.configure(text="开始比较")
            self.show_info("")  # 清空提示信息

    def start_comparison(self):
        """开始图像比较"""
        try:
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
                alpha = self.alpha_scale.get() / 100.0
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

            # 显示结果
            if self.left_result is not None:
                self.display_image(self.left_canvas, self.left_result, self.left_markers)
            if self.right_result is not None:
                self.display_image(self.right_canvas, self.right_result, self.right_markers)

            # 根据比较结果更新提示信息
            if self.left_result is None:
                self.show_info("OCR识别失败")
                print("OCR识别失败")
            else:
                # 检查是否有差异（通过检查图像是否被修改）
                if np.array_equal(self.left_result, self.left_image if mode == "left" else self.right_image):  # type: ignore
                    self.show_info("OCR结果相同")
                    print("OCR结果相同")
                else:
                    self.show_info("已标记出差异区域")

        except Exception as e:
            print(e)
            self.show_info(f"比较出错: {str(e)}")
            print(e)
            self.is_comparing = False
            self.compare_button.config(text="开始比较")

    def load_image(self, side):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not file_path:
            return

        image = cv2.imread(file_path)  # pylint: disable=no-member
        if image is None:
            self.show_info(f"无法加载图片: {file_path}")
            return

        if side == "left":
            self.left_image = image
            self.left_result = None  # 清除之前的比较结果
            self.show_info("已加载左图")
        else:
            self.right_image = image
            self.right_result = None  # 清除之前的比较结果
            self.show_info("已加载右图")

        # 如果正在比较状态，切换回原图状态
        if self.is_comparing:
            self.is_comparing = False
            self.compare_button.configure(text="开始比较")

        # 显示图像
        if side == "left":
            self.display_image(self.left_canvas, self.left_image, self.left_markers)
        else:
            self.display_image(self.right_canvas, self.right_image, self.right_markers)

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

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()