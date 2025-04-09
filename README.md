
比较两张图的差异，主要是已经转成图的图书文档页面。

目前有三种比较模式：像素比较；叠置比较；OCR文字比较(图上显示差异 + diff HTML报告)。

## 键盘快捷键及其功能

|快捷键 | 功能（默认对A图，加Ctrl后同时针对A、B图）|
| ---     | ---      |
| Ctrl+P | 比较/原图 |
| PgUp | 上翻页 |
| PgDn    | 下翻页   |
| Left    | 左移     |
| Right   | 右移     |
| Up      | 上移     |
| Down    | 下移     |
| Shift + | 放大     |
| Shift - | 缩小     |
| tab     | 切换图层 |

## TODO

* 尝试：OpenCV的subtract、absdiff、加、减；结构相似性指数(SSIM)方法，使用 scikit-image 的 skimage.metrics.structural_similarity
* 增加：活文字PDF的比较
* 序列图、PDF的比较

## 资料

* [imageMagick](https://stackoverflow.com/questions/5132749/diff-an-image-using-imagemagick)
* Beyond Compare
* Adobe Acrobat Pro
