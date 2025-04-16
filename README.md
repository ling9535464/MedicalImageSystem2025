

---

# 项目 README
image_dataset：（不包含分割与分期分级数据集）
通过网盘分享的文件：2021-2023all等2个文件
链接: https://pan.baidu.com/s/1JzMBOwV-Wi39mmqsCK5tTg?pwd=btps 提取码: btps

模拟用户使用的测试图片：
通过网盘分享的文件：验证集1 - 所有图片等2个文件
链接: https://pan.baidu.com/s/16sCCJ4qqtiE1EV17fmvFRw?pwd=psbw 提取码: psbw

分割模块所需预训练模型：
通过网盘分享的文件：sam_vit_b_01ec64.pth
链接: https://pan.baidu.com/s/1J7zhRMd3oazKxW56W3jf9Q?pwd=af6x 提取码: af6x

## 项目概述
本项目包含多个模块，用于医学图像处理和分析。这些模块涵盖了从图像分类、分割到异常检测和分级的任务。以下是每个模块的简要介绍：

1. **良恶性分类模块**：
   - `1classify（多进程）.py`：训练用于良恶性分类的模型。
   - `1test.py`：测试已训练的分类模型。

2. **肿瘤分割模块**：
   - `2polygon2mask.py`：从标注图像中提取掩码，用于训练分割模型。
   - `2finetune_medsam copy.py`：微调 SAM 模型以进行医学图像分割。
   - `2run_medsam_anomaly.py`：运行微调后的 SAM 模型进行异常区域分割。

3. **肿瘤分级分期模块**：
   - `3grading.py`：实现肿瘤分级分期的一部分代码。

## 模块 1：良恶性分类模块

### 功能
- 使用 EfficientNet 模型对医学图像进行分类，判断图像是否包含异常（良性或恶性）。
- 支持 K 折交叉验证、数据增强和早停机制。

### 使用方法
1. 确保安装了以下依赖：
   - `torch`
   - `torchvision`
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `tqdm`
   - `Pillow`
   - `imagehash`

2. 配置路径：
   - `image_dataset/用于训练的所有图片`：包含所有训练图像的文件夹。
   - `image_dataset/用于训练的异常图片`：包含异常图像的文件夹。

3. 训练模型：
   ```bash
   python 1classify（多进程）.py
   ```

4. 测试模型：
   ```bash
   python 1test.py
   ```

### 输出
- 分类模型保存在 `classify/best_model_final.pth`。
- K 折交叉验证结果保存在 `classify/kfold_results.csv`。
- 测试结果保存在 `错误分类图片.csv` 和 `part1_pr_curve.png`。

## 模块 2：肿瘤分割模块

### 功能
- **2polygon2mask.py**：从标注图像中提取掩码，用于训练分割模型。
- **2finetune_medsam copy.py**：微调 SAM 模型以进行医学图像分割。
- **2run_medsam_anomaly.py**：运行微调后的 SAM 模型进行异常区域分割。

### 使用方法
1. 确保安装了以下依赖：
   - `torch`
   - `torchvision`
   - `numpy`
   - `opencv-python`
   - `matplotlib`
   - `segment-anything`

2. 配置路径：
   - `image_dataset/分割/原图`：包含原始图像的文件夹。
   - `image_dataset/分割/标注分割`：包含标注图像的文件夹。

3. 生成掩码：
   ```bash
   python 2polygon2mask.py
   ```

4. 微调 SAM 模型：
   ```bash
   python 2finetune_medsam copy.py
   ```

5. 运行分割模型：
   ```bash
   python 2run_medsam_anomaly.py
   ```

### 输出
- 生成的掩码保存在 `掩码分割` 文件夹中。
- 分割模型保存在 `logs/best_model_final.pth`。
- 分割结果保存在 `results` 文件夹中。

## 模块 3：肿瘤分级分期模块

### 功能
- 使用 EfficientNet 模型对医学图像进行分级，判断异常的严重程度。
- 支持 K 折交叉验证和早停机制。

### 使用方法
1. 确保安装了以下依赖：
   - `torch`
   - `torchvision`
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `tqdm`
   - `Pillow`

2. 配置路径：
   - `image_dataset/分期分级/分级部分数据.xlsx`：包含图像分级信息的 Excel 文件。
   - `image_dataset/用于训练的异常图片`：包含异常图像的文件夹。

3. 运行脚本：
   ```bash
   python 3grading.py
   ```

### 输出
- 分级模型保存在 `grading/best_model_final.pth`。
- K 折交叉验证结果保存在 `grading/kfold_results.csv`。

## 项目结构
```
project/
├── 1classify（多进程）.py
├── 1test.py
├── 2polygon2mask.py
├── 2finetune_medsam copy.py
├── 2run_medsam_anomaly.py
├── 3grading.py
├── image_dataset/
│   ├── 用于训练的所有图片/
│   ├── 用于训练的异常图片/
│   ├── 分割/
│   │   ├── 原图/
│   │   ├── 标注分割/
│   │   ├── 掩码分割/
│   ├── 分期分级/
│   │   ├── 分级部分数据.xlsx
├── classify/
│   ├── best_model_final.pth
│   ├── kfold_results.csv
├── logs/
│   ├── best_model_final.pth
├── results/
│   ├── 分割结果.png
├── grading/
│   ├── best_model_final.pth
│   ├── kfold_results.csv
```



---

