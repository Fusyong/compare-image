import tkinter as tk
from tkinter import ttk, filedialog
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

        # 图像数据
        self.left_image = None
        self.right_image = None
        self.left_markers = [(5, 5), (95, 95)]  # L1, L2 (L2在右下角-5的位置)
        self.right_markers = [(5, 5), (95, 95)]  # R1, R2 (R2在右下角-5的位置)
        self.active_marker = None

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

        ttk.Button(toolbar, text="加载左图", command=lambda: self.load_image("left")).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(toolbar, text="加载右图", command=lambda: self.load_image("right")).pack(fill=tk.X, padx=5, pady=2)

        # 添加模式选择
        mode_frame = ttk.Frame(toolbar)
        mode_frame.pack(fill=tk.X, padx=5, pady=2)
        self.mode_var = tk.StringVar(value="compare")
        ttk.Radiobutton(mode_frame, text="比较模式", variable=self.mode_var, value="compare").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="叠加模式", variable=self.mode_var, value="overlay").pack(side=tk.LEFT)

        # 叠加模式的透明度控制
        alpha_frame = ttk.Frame(toolbar)
        alpha_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(alpha_frame, text="透明度:").pack(side=tk.LEFT)
        self.alpha_scale = ttk.Scale(alpha_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.alpha_scale.set(50)  # 默认透明度0.5
        self.alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.compare_button = ttk.Button(toolbar, text="开始比较", command=self.toggle_compare)
        self.compare_button.pack(fill=tk.X, padx=5, pady=2)

        # 坐标输入区域（移到左侧）
        coords_frame = ttk.LabelFrame(left_panel, text="标记点坐标")
        coords_frame.pack(fill=tk.X, pady=5)

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

    def toggle_compare(self):
        """切换比较/原图显示状态"""
        if self.left_image is None or self.right_image is None:
            return

        self.is_comparing = not self.is_comparing
        mode = self.mode_var.get()

        if self.is_comparing:
            self.compare_button.configure(text="显示原图")
            # 根据模式选择处理方法
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
            else:  # overlay mode
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

            # 显示结果
            if self.left_result is not None:
                self.display_image(self.left_canvas, self.left_result, self.left_markers)
            if self.right_result is not None:
                self.display_image(self.right_canvas, self.right_result, self.right_markers)
        else:
            self.compare_button.configure(text="开始比较")
            # 显示原图
            if self.left_image is not None:
                self.display_image(self.left_canvas, self.left_image, self.left_markers)
            if self.right_image is not None:
                self.display_image(self.right_canvas, self.right_image, self.right_markers)

    def load_image(self, side):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            return

        if side == "left":
            self.left_image = image
            self.left_result = None  # 清除之前的比较结果
        else:
            self.right_image = image
            self.right_result = None  # 清除之前的比较结果

        # 如果正在比较状态，切换回原图状态
        if self.is_comparing:
            self.is_comparing = False
            self.compare_button.configure(text="比较图像")

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

        image = cv2.resize(image, (canvas_width, canvas_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 绘制标记点
        for i, (x, y) in enumerate(markers):
            x = int(x * canvas_width / 100)
            y = int(y * canvas_height / 100)
            cv2.drawMarker(image, (x, y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(image, f"{'L' if canvas == self.left_canvas else 'R'}{i+1}",
                       (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 显示图像
        photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # 保持引用

    def start_drag(self, event, side):
        canvas = self.left_canvas if side == "left" else self.right_canvas
        markers = self.left_markers if side == "left" else self.right_markers

        # 检查是否点击了标记点
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        x_rel = event.x / canvas_width * 100
        y_rel = event.y / canvas_height * 100

        for i, (mx, my) in enumerate(markers):
            if abs(x_rel - mx) < 5 and abs(y_rel - my) < 5:
                self.active_marker = (side, i)
                break

    def drag(self, event, side):
        if not self.active_marker or self.active_marker[0] != side:
            return

        canvas = self.left_canvas if side == "left" else self.right_canvas
        markers = self.left_markers if side == "left" else self.right_markers

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

        # 更新坐标输入框
        self.update_coordinate_entries()

    def end_drag(self, event):
        self.active_marker = None

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

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()