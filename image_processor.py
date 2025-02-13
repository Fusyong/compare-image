import json
import cv2  # type: ignore
import numpy as np
import difflib  # 新增：引入 difflib 模块，用于文本比较
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
        scaled_image = cv2.resize(image, (new_width, new_height))  # pylint: disable=no-member

        # 计算缩放后的点坐标
        scaled_points = []
        for x, y in src_points:
            scaled_points.append((x * scale, y * scale))

        return scaled_image, scaled_points

    def get_transform_matrix(self, src_point, dst_point, angle=0):
        """计算平移变换矩阵"""
        # 创建平移矩阵
        dx = dst_point[0] - src_point[0]
        dy = dst_point[1] - src_point[1]

        # 使用浮点数数组创建变换矩阵
        matrix = np.array([
            [1.0, 0.0, dx],
            [0.0, 1.0, dy]
        ], dtype=np.float32)

        return matrix

    def transform_image(self, image, matrix, target_size):
        """应用变换矩阵到图像"""
        return cv2.warpAffine(image, matrix, target_size)  # pylint: disable=no-member

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
            print(e)
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
            gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member

            # 创建白色区域的掩码
            _, white_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)  # pylint: disable=no-member

            # 对掩码进行平滑处理
            white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)  # pylint: disable=no-member

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
            print(e)
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
            base_b, base_g, base_r = cv2.split(base_image)  # pylint: disable=no-member
            aligned_b, aligned_g, aligned_r = cv2.split(aligned_image)  # pylint: disable=no-member

            # 使用视觉敏感度权重转换为灰度图
            base_gray = 0.114*base_b + 0.587*base_g + 0.299*base_r
            aligned_gray = 0.114*aligned_b + 0.587*aligned_g + 0.299*aligned_r

            # 计算两张图的差异，用差异生成灰度的结果图（白色表示相同，黑色表示不同）
            diff = cv2.absdiff(base_gray, aligned_gray)  # pylint: disable=no-member
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

    def compare_ocr_results(self, base_result, counter_result, scale, dx, dy):
        """比较两个OCR结果，返回无法匹配的文本框

        Args:
            base_result: 基准图OCR结果
            counter_result: 对比图OCR结果
            scale: 缩放比例
            dx: x方向平移量
            dy: y方向平移量

        Returns:
            unmatched: 未匹配的文本框列表，每项包含 (box, text)
        """
        unmatched = []
        # 预先计算对比图中每个文本框经过缩放和平移后的坐标，避免重复计算
        scaled_counter_result = []
        for counter_item in counter_result:
            text = counter_item[1]
            scaled_box = np.array([[p[0] * scale + dx, p[1] * scale + dy] for p in counter_item[0]])
            scaled_counter_result.append((text, scaled_box))

        # # 将ndarray转换为列表以便JSON序列化
        # json_data = []
        # for text, box in scaled_counter_result:
        #     json_data.append({
        #         "text": text,
        #         "box": box.tolist()  # 将ndarray转换为列表
        #     })
        # with open("scaled_counter_result.json", "w", encoding='utf-8') as f:
        #     json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 遍历基准图OCR结果，检查是否存在匹配的文本框
        for item in base_result:
            text = item[1]
            found = False
            for cand_text, cand_box in scaled_counter_result:
                if cand_text == text:
                    overlap_ratio = self.calculate_overlap_ratio(cand_box, item[0])
                    if overlap_ratio > 0.3:
                        found = True
                        break
            if not found:
                unmatched.append(item)
        return unmatched

    def ocr_and_compare(self, left_image, right_image, left_markers, right_markers, mode="left"):
        """OCR两张图片并比较差异，返回标记了差异的图片"""

        # # 如果是完全相同的图像，直接返回原图
        # if left_image.shape == right_image.shape and np.array_equal(left_image, right_image):
        #     self.show_info("两侧图像相同")
        #     return left_image if mode == "left" else right_image

        try:
            # 初始化OCR引擎
            engine = RapidOCR()

            # 检查是否需要重新OCR
            if self.need_ocr(left_image, "left"):
                self.show_info("正在OCR左图...")
                left_result, _ = engine(left_image, return_word_box=True)
                if left_result:
                    self.cached_left_result = left_result
                    # with open("left_result.json", "w", encoding='utf-8') as f:
                    #     json.dump(left_result, f, ensure_ascii=False, indent=2)
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
                    # with open("right_result.json", "w", encoding='utf-8') as f:
                    #     json.dump(right_result, f, ensure_ascii=False, indent=2)
                else:
                    self.show_info("右图OCR识别失败")
                    return None
            else:
                right_result = self.cached_right_result
                self.show_info("使用右图缓存的OCR结果")

            # ----------------- 新增OCR文本比较功能 -----------------
            # 根据 mode 确定基准OCR文本和对比OCR文本
            if mode == "left":
                base_text = "\n".join([item[1] for item in left_result])
                compare_text = "\n".join([item[1] for item in right_result])
            else:
                base_text = "\n".join([item[1] for item in right_result])
                compare_text = "\n".join([item[1] for item in left_result])
            diff_lines = list(difflib.unified_diff(
                base_text.splitlines(),
                compare_text.splitlines(),
                fromfile='Base OCR Text',
                tofile='Compare OCR Text',
                lineterm=''
            ))
            diff_text = "\n".join(diff_lines)
            if diff_text:
                self.show_info("OCR文本差异：\n" + diff_text)
            else:
                self.show_info("OCR文本无差异")
            # ----------------- OCR文本比较功能结束 -----------------

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

            # 比较OCR结果
            self.show_info("比较OCR结果...")
            unmatched = self.compare_ocr_results(
                left_result if mode == "left" else right_result,
                right_result if mode == "left" else left_result,
                scale, dx, dy
            )

            has_difference = False

            # 根据模式标记未匹配的文本框
            if unmatched:
                has_difference = True
                for box, text, *_ in unmatched:
                    # 将文本框坐标转换为整数
                    box = np.array(box, dtype=np.int32).reshape((-1, 2))
                    # 绘制红色细线边框
                    cv2.polylines(result, [box], True, (0, 0, 255), 1)  # pylint: disable=no-member
                    # # 在文本框上方添加文字标记
                    # text_pos = (int(min(p[0] for p in box)), int(min(p[1] for p in box)) - 5)
                    # # 使用PIL绘制灰色文字
                    # img_pil = Image.fromarray(result)
                    # draw = ImageDraw.Draw(img_pil)
                    # font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 9)
                    # draw.text(text_pos, text, font=font, fill=(128, 128, 128))
                    # result = np.array(img_pil)

            if not has_difference:
                self.show_info("OCR结果相同")

            return result
        except Exception as e:
            self.show_info(f"OCR比较出错: {str(e)}")
            print("OCR错误:", e)
            return None

    def calculate_overlap_ratio(self, box1, box2):
        """计算两个文本框的重叠比例"""
        try:
            # 将文本框坐标转换为浮点型的numpy数组
            poly1 = np.array(box1, dtype=np.float32).reshape((-1, 2))
            poly2 = np.array(box2, dtype=np.float32).reshape((-1, 2))

            # 计算各自的面积
            area1 = cv2.contourArea(poly1)  # pylint: disable=no-member
            area2 = cv2.contourArea(poly2)  # pylint: disable=no-member

            # 计算两个凸多边形的交集面积
            retval, intersection = cv2.intersectConvexConvex(poly1, poly2)  # pylint: disable=no-member
            intersection_area = retval

            # 计算并集面积
            union_area = area1 + area2 - intersection_area
            if union_area == 0:
                return 0
            return intersection_area / union_area
        except Exception as e:
            self.show_info(f"计算重叠比例出错: {str(e)}")
            return 0

