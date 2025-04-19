
一个用于比较两组图像差异的 Python 工具，特别适用于图书文档页面的比较。该工具提供了多种比较模式，可以帮助用户快速发现图像之间的差异。

## 安装

```bash
pip3 install opencv-python
# OCR文字比较的依赖
pip install rapidocr-onnxruntime
```

## 功能与使用

- **运行程序**：
   ```bash
   python main.py
   ```

- **加载图像**：
   - 默认从`img/L`和`img/R`路径自动加载左右图组
   - 点击"加载左图"和"加载右图"按钮选择要比较的图像
   - 支持批量加载图像组

- **多种比较模式**：
  - 像素比较：直接比较两张图片的像素差异
  - 叠置比较：将两张图片叠加显示，支持透明度调节
  - OCR文字比较：识别并比较图片中的文字内容，生成差异报告

- **图像处理功能**：
  - 支持图像缩放和对齐
  - 根据标记点缩放与对齐
    - 使用鼠标左键拖拽标记点
    - 左两点与右两点用作缩放、对齐的根据
    - 1、2点之间的距离尽可能远，以便准确缩放
  - 图像缓存机制，提高处理效率

- **用户界面特性**：
  - 区域放大功能，方便查看细节
  - 快捷键支持
    - `Alt + Q`：在原图与比较间切换
    - `Alt + Z`：开关区域放大模式
    - `Ctrl + left mouse`：区域放大
    - `PgUp`: 前一对图像
    - `PgDn`: 后一对图像
  - 图像导航功能

## 技术实现

- 使用 OpenCV 进行图像处理
- 使用 RapidOCR 进行文字识别
- 使用 Tkinter 构建图形界面
- 使用 NumPy 进行数值计算

## TODO

- 尝试更多图像比较算法：
  - OpenCV 的 subtract、absdiff 等方法
  - 结构相似性指数(SSIM)方法
  - 使用 scikit-image 的 skimage.metrics.structural_similarity
- 增加对活文字 PDF 的支持

## 参考资料

- [ImageMagick](https://stackoverflow.com/questions/5132749/diff-an-image-using-imagemagick)
- Beyond Compare
- Adobe Acrobat Pro
