import json
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageDraw, ImageFont

class ImageProcessor:
    def __init__(self):
        self.info_callback = None  # 用于显示信息的回调函数
        # 添加缓存相关的属性
        self.left_image_hash = None  # 左图像的哈希值
        self.right_image_hash = None  # 右图像的哈希值
        self.cached_left_result = None  # 缓存的左图OCR结果
        self.cached_right_result = None  # 缓存的右图OCR结果

    def set_info_callback(self, callback):
        """设置信息显示回调函数"""
        self.info_callback = callback

    def show_info(self, message):
        """显示信息"""
        if self.info_callback:
            self.info_callback(message)
        else:
            print(message)  # 如果没有设置回调函数，则使用print作为后备方案

    def get_distance(self, p1, p2):
        """计算两点之间的距离"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def normalize_scale(self, image, src_points, dst_points):
        """根据两对点的距离调整图像比例"""
        # 计算源图和目标图中两点的距离
        src_dist = self.get_distance(src_points[0], src_points[1])
        dst_dist = self.get_distance(dst_points[0], dst_points[1])

        # 计算缩放比例
        scale = dst_dist / src_dist

        # 计算新的图像尺寸
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 缩放图像
        scaled_image = cv2.resize(image, (new_width, new_height))

        # 计算缩放后的点坐标
        scaled_points = []
        for x, y in src_points:
            scaled_points.append((int(x * scale), int(y * scale)))

        return scaled_image, scaled_points

    def get_transform_matrix(self, src_point, dst_point, angle=0):
        """计算平移变换矩阵"""
        # 创建平移矩阵
        dx = dst_point[0] - src_point[0]
        dy = dst_point[1] - src_point[1]

        matrix = np.float32([
            [1, 0, dx],
            [0, 1, dy]
        ])
        return matrix

    def transform_image(self, image, matrix, target_size):
        """应用变换矩阵到图像"""
        return cv2.warpAffine(image, matrix, target_size)

    def normalize_coordinates(self, markers, image_shape):
        """将相对坐标（百分比）转换为绝对坐标"""
        height, width = image_shape[:2]
        return [(int(x * width / 100), int(y * height / 100)) for x, y in markers]

    def preprocess_images(self, left_image, right_image, left_markers, right_markers, mode="left"):
        """预处理两张图片，包括标准化坐标、缩放和对齐

        Args:
            left_image: 左图像
            right_image: 右图像
            left_markers: 左图像上的标记点 [(x1, y1), (x2, y2)]
            right_markers: 右图像上的标记点 [(x1, y1), (x2, y2)]
            mode: 处理模式，"left" 或 "right"，决定以哪张图为基准

        Returns:
            (base_image, aligned_image) 处理后的基准图像和对齐后的移动图像
        """
        try:
            # 转换标记点坐标从相对坐标（百分比）到绝对坐标
            left_points = self.normalize_coordinates(left_markers, left_image.shape)
            right_points = self.normalize_coordinates(right_markers, right_image.shape)

            if mode == "left":
                # 以左图为基准
                base_image = left_image.copy()
                base_points = left_points
                moving_image = right_image.copy()
                moving_points = right_points
            else:
                # 以右图为基准
                base_image = right_image.copy()
                base_points = right_points
                moving_image = left_image.copy()
                moving_points = left_points

            # 1. 先调整moving_image的缩放比例，使两对标记点距离相等
            scaled_image, scaled_points = self.normalize_scale(
                moving_image, moving_points, base_points
            )

            # 2. 基于第一个点（L1/R1）进行对齐
            matrix = self.get_transform_matrix(
                scaled_points[0],  # 移动图像上的第一个点
                base_points[0]     # 基准图像上的第一个点
            )

            # 3. 对齐图像
            height, width = base_image.shape[:2]
            aligned_image = self.transform_image(scaled_image, matrix, (width, height))

            return base_image, aligned_image
        except Exception as e:
            self.show_info(f"图像预处理出错: {str(e)}")
            return None, None

    def overlay_images(self, left_image, right_image, left_markers, right_markers, mode="left", alpha=0.5):
        """叠加两张图片"""
        try:
            # 预处理图像
            base_image, aligned_image = self.preprocess_images(
                left_image, right_image, left_markers, right_markers, mode
            )
            if base_image is None or aligned_image is None:
                return None

            # 处理moving_image的白色区域
            # 转换为灰度图
            gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)

            # 创建白色区域的掩码
            _, white_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

            # 对掩码进行平滑处理
            white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)

            # 将掩码转换为0-1范围的浮点数
            white_mask = white_mask.astype(float) / 255.0

            # 创建透明度矩阵
            alpha_matrix = np.ones_like(white_mask) * alpha
            # 白色区域的透明度设为1（完全透明）
            alpha_matrix = alpha_matrix * (1.0 - white_mask)

            # 扩展alpha_matrix为3通道
            alpha_matrix = np.stack([alpha_matrix] * 3, axis=-1)

            # 叠加图像
            result = base_image.copy()
            mask = (1.0 - alpha_matrix)
            result = (result * mask + aligned_image * alpha_matrix).astype(np.uint8)

            return result
        except Exception as e:
            self.show_info(f"图像叠加出错: {str(e)}")
            return None

    def compare_images(self, left_image, right_image, left_markers, right_markers, mode="left"):
        """比较两张图片的差异，直接展示像素级的视觉差异"""
        try:
            # 预处理图像
            base_image, aligned_image = self.preprocess_images(
                left_image, right_image, left_markers, right_markers, mode
            )
            if base_image is None or aligned_image is None:
                return None

            # 把两张图片按视觉敏感度转换为灰度图（越敏感的颜色越亮）
            base_b, base_g, base_r = cv2.split(base_image)
            aligned_b, aligned_g, aligned_r = cv2.split(aligned_image)

            # 使用视觉敏感度权重转换为灰度图
            base_gray = 0.114*base_b + 0.587*base_g + 0.299*base_r
            aligned_gray = 0.114*aligned_b + 0.587*aligned_g + 0.299*aligned_r

            # 计算两张图的差异，用差异生成灰度的结果图（白色表示相同，黑色表示不同）
            diff = cv2.absdiff(base_gray, aligned_gray)
            result = 255 - diff.astype(np.uint8)  # 反转差异值，使相同区域显示为白色

            return result
        except Exception as e:
            self.show_info(f"图像处理出错: {str(e)}")
            return None

    def calculate_image_hash(self, image):
        """计算图像的哈希值用于判断图像是否改变"""
        return hash(image.tobytes())

    def need_ocr(self, image, side="left"):
        """判断是否需要重新进行OCR"""
        current_hash = self.calculate_image_hash(image)
        if side == "left":
            if current_hash != self.left_image_hash:
                self.left_image_hash = current_hash
                return True
        else:
            if current_hash != self.right_image_hash:
                self.right_image_hash = current_hash
                return True
        return False

    def ocr_and_compare(self, left_image, right_image, left_markers, right_markers, mode="left"):
        """OCR两张图片并比较差异，返回标记了差异的图片"""

        # 如果是完全相同的图像，直接返回原图
        if left_image.shape == right_image.shape and np.array_equal(left_image, right_image):
            self.show_info("两侧图像相同")
            return left_image if mode == "left" else right_image

        try:
            # 初始化OCR引擎
            engine = RapidOCR()

            # 检查是否需要重新OCR
            if self.need_ocr(left_image, "left"):
                self.show_info("正在OCR左图...")
                left_result, _ = engine(left_image, return_word_box=True)
                if left_result:
                    self.cached_left_result = left_result
                    with open("left_result.json", "w", encoding='utf-8') as f:
                        json.dump(left_result, f, ensure_ascii=False, indent=2)
                else:
                    self.show_info("左图OCR识别失败")
                    return None
            else:
                left_result = self.cached_left_result
                self.show_info("使用左图缓存的OCR结果")

            if self.need_ocr(right_image, "right"):
                self.show_info("正在OCR右图...")
                right_result, _ = engine(right_image, return_word_box=True)
                if right_result:
                    self.cached_right_result = right_result
                    with open("right_result.json", "w", encoding='utf-8') as f:
                        json.dump(right_result, f, ensure_ascii=False, indent=2)
                else:
                    self.show_info("右图OCR识别失败")
                    return None
            else:
                right_result = self.cached_right_result
                self.show_info("使用右图缓存的OCR结果")

            # 获取对齐后的图像用于显示结果
            base_image, aligned_image = self.preprocess_images(
                left_image, right_image, left_markers, right_markers, mode
            )
            if base_image is None or aligned_image is None:
                self.show_info("图像对齐失败")
                return None

            # 创建结果图像
            result = base_image.copy()

            # 计算变换参数
            src_points = self.normalize_coordinates(
                right_markers if mode == "left" else left_markers,
                right_image.shape if mode == "left" else left_image.shape
            )
            dst_points = self.normalize_coordinates(
                left_markers if mode == "left" else right_markers,
                left_image.shape if mode == "left" else right_image.shape
            )

            # 计算缩放比例
            src_dist = self.get_distance(src_points[0], src_points[1])
            dst_dist = self.get_distance(dst_points[0], dst_points[1])
            scale = dst_dist / src_dist

            # 计算平移量
            dx = dst_points[0][0] - src_points[0][0] * scale
            dy = dst_points[0][1] - src_points[0][1] * scale

            # 用于存储已匹配的文本框
            matched_boxes_left = set()
            matched_boxes_right = set()
            has_difference = False

            # 首先尝试通过文本内容匹配
            text_to_boxes_right = {}
            for i, item in enumerate(right_result):
                text = item[1]
                if text not in text_to_boxes_right:
                    text_to_boxes_right[text] = []
                text_to_boxes_right[text].append((i, item[0]))

            # 遍历左侧图像中的每个文本框
            for i, left_item in enumerate(left_result):
                left_box = left_item[0]  # 文本框坐标
                left_text = left_item[1]  # 文本内容

                found_match = False
                # 先查找相同文本的候选框
                if left_text in text_to_boxes_right:
                    candidates = text_to_boxes_right[left_text]
                    best_overlap = 0
                    best_match = None

                    for j, right_box in candidates:
                        if j in matched_boxes_right:
                            continue

                        # 变换右侧文本框坐标
                        transformed_box = []
                        for x, y in right_box:
                            # 应用缩放和平移变换
                            tx = x * scale + dx
                            ty = y * scale + dy
                            transformed_box.append([tx, ty])

                        # 计算重叠度
                        overlap_ratio = self.calculate_overlap_ratio(left_box, transformed_box)

                        # 记录最佳匹配
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_match = (j, transformed_box)

                    # 如果找到足够好的匹配
                    if best_match and best_overlap > 0.3:  # 降低阈值以适应坐标变换的误差
                        found_match = True
                        matched_boxes_left.add(i)
                        matched_boxes_right.add(best_match[0])

                # 如果没有找到匹配，在左侧图像上标记差异
                if not found_match and mode == "left":
                    has_difference = True
                    # 将文本框坐标转换为整数
                    box = np.array(left_box, dtype=np.int32).reshape((-1, 2))
                    # 绘制半透明黄色边框
                    cv2.polylines(result, [box], True, (0, 255, 255), 2)
                    # 在文本框上方添加文字标记（使用宋体6号字，约10.5pt或14px）
                    text_pos = (int(min(p[0] for p in left_box)), int(min(p[1] for p in left_box)) - 5)
                    # 创建宋体字体
                    font_path = "C:/Windows/Fonts/simsun.ttc"  # 宋体字体路径
                    font_size = 14
                    img_pil = Image.fromarray(result)
                    draw = ImageDraw.Draw(img_pil)
                    font = ImageFont.truetype(font_path, font_size)
                    # 绘制灰色文字
                    draw.text(text_pos, left_text, font=font, fill=(128, 128, 128))
                    # 转换回OpenCV格式
                    result = np.array(img_pil)

            # 标记右侧图像中未匹配的文本框
            if mode == "right":
                for j, right_item in enumerate(right_result):
                    if j not in matched_boxes_right:
                        has_difference = True
                        right_box = right_item[0]
                        right_text = right_item[1]

                        # 变换文本框坐标
                        transformed_box = []
                        for x, y in right_box:
                            # 应用缩放和平移变换
                            tx = x * scale + dx
                            ty = y * scale + dy
                            transformed_box.append([tx, ty])

                        # 将文本框坐标转换为整数
                        box = np.array(transformed_box, dtype=np.int32).reshape((-1, 2))
                        # 绘制半透明黄色边框
                        cv2.polylines(result, [box], True, (0, 255, 255), 2)
                        # 在文本框上方添加文字标记
                        text_pos = (int(min(p[0] for p in transformed_box)), int(min(p[1] for p in transformed_box)) - 5)
                        # 使用PIL绘制灰色文字
                        img_pil = Image.fromarray(result)
                        draw = ImageDraw.Draw(img_pil)
                        font = ImageFont.truetype(font_path, font_size)
                        draw.text(text_pos, right_text, font=font, fill=(128, 128, 128))
                        result = np.array(img_pil)

            if not has_difference:
                self.show_info("OCR结果相同")

            return result
        except Exception as e:
            self.show_info(f"OCR比较出错: {str(e)}")
            return None

    def calculate_overlap_ratio(self, box1, box2):
        """计算两个文本框的重叠比例"""
        try:
            # 将文本框坐标转换为numpy数组
            box1 = np.array(box1, dtype=np.int32).reshape((-1, 2))
            box2 = np.array(box2, dtype=np.int32).reshape((-1, 2))

            # 创建掩码图像
            mask1 = np.zeros((1000, 1000), dtype=np.uint8)
            mask2 = np.zeros((1000, 1000), dtype=np.uint8)

            # 在掩码上填充文本框
            cv2.fillPoly(mask1, [box1], (1))
            cv2.fillPoly(mask2, [box2], (1))

            # 计算交集面积
            intersection = cv2.bitwise_and(mask1, mask2)
            intersection_area = cv2.countNonZero(intersection)

            # 计算并集面积
            union = cv2.bitwise_or(mask1, mask2)
            union_area = cv2.countNonZero(union)

            # 计算重叠比例
            if union_area == 0:
                return 0
            return intersection_area / union_area
        except Exception as e:
            self.show_info(f"计算重叠比例出错: {str(e)}")
            return 0

