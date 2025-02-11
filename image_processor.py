import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

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
            print(f"图像预处理出错: {str(e)}")
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
            print(f"图像叠加出错: {str(e)}")
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
            print(f"图像处理出错: {str(e)}")
            return None