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

    def compare_images(self, left_image, right_image, left_markers, right_markers, mode="left"):
        """比较两张图片的差异

        Args:
            left_image: 左图像
            right_image: 右图像
            left_markers: 左图像上的标记点 [(x1, y1), (x2, y2)]
            right_markers: 右图像上的标记点 [(x1, y1), (x2, y2)]
            mode: 比较模式，"left" 或 "right"，决定以哪张图为基准

        Returns:
            比较结果图像
        """
        try:
            # 转换标记点坐标从相对坐标（百分比）到绝对坐标
            left_points = self.normalize_coordinates(left_markers, left_image.shape)
            right_points = self.normalize_coordinates(right_markers, right_image.shape)

            if mode == "left":
                # 以左图为基准
                base_image = left_image
                base_points = left_points
                moving_image = right_image
                moving_points = right_points
            else:
                # 以右图为基准
                base_image = right_image
                base_points = right_points
                moving_image = left_image
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

            # 4. 计算差异
            diff = cv2.absdiff(base_image, aligned_image)

            # 5. 创建掩码：相同像素为白色，不同像素保持原样
            mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # 6. 将相同的区域设为白色，不同的区域保持原样
            result = cv2.bitwise_and(base_image, mask)
            white = np.ones_like(result) * 255
            result = cv2.bitwise_or(result, cv2.bitwise_and(white, cv2.bitwise_not(mask)))

            return result
        except Exception as e:
            print(f"图像处理出错: {str(e)}")
            return None