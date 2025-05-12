

---

# 项目 README
项目所需数据集与SAM预训练模型：
通过网盘分享的文件：项目所需数据集与SAM预训练模型(1).zip
链接: https://pan.baidu.com/s/1OD7fEcIYB-sr2yWF_lqNJg?pwd=cu6s 提取码: cu6s 

---

## 项目概述
本项目包含多个模块，用于医学图像的处理和分析，涵盖图像分类、分割、异常检测以及分级分期等任务。以下是每个模块的简要介绍：

1. **图像分类模块**：
   - `1classify.py`：训练用于图像分类的模型，判断图像是否包含异常（良性或恶性）。（建议使用multiproceed版本）
   - `1test.py`：测试已训练的分类模型，评估其在测试集上的性能。

2. **图像分割模块**：
   - `2polygon2mask.py`：从标注图像中提取掩码，用于训练分割模型。
   - `2finetune_medsam.py`：微调 SAM 模型以进行医学图像分割。
   - `2run_medsam_anomaly.py`：运行微调后的 SAM 模型进行异常区域分割。

3. **分级分期模块**：
   - `3grading.py`：实现肿瘤分级与分子分型的代码，用于判断异常的严重程度。
   - `3test.py`：测试训练出的分级模型，评估其在测试集上的性能。

## 模块 1：图像分类模块

### 功能
- 使用 EfficientNet 模型对医学图像进行分类，判断图像是否包含异常（良性或恶性）。
- 支持 K 折交叉验证、数据增强、加权采样和早停机制。
- 提供详细的训练和验证指标可视化。

### 使用方法
1. **依赖安装**：
   确保安装了以下依赖：
   - `torch`
   - `torchvision`
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `tqdm`
   - `Pillow`
   - `matplotlib`

2. **路径配置**：
   - `image_dataset/用于训练的所有图片`：包含所有训练图像的文件夹。
   - `image_dataset/用于训练的异常图片`：包含异常图像的文件夹。

3. **训练模型**：
   ```bash
   python 1classify.py
   ```

4. **测试模型**：
   ```bash
   python 1test.py
   ```

### 输出
- 分类模型保存在 `classify/best_model_final.pth`。
- K 折交叉验证结果保存在 `classify/kfold_results.csv`。
- 测试结果保存在 `valset/valset_所有图片分类结果.csv`。
- 训练过程的可视化图表保存在 `classify/` 文件夹中。

## 模块 2：图像分割模块

### 功能
- **2polygon2mask.py**：从标注图像中提取掩码，用于训练分割模型。
- **2finetune_medsam.py**：微调 SAM 模型以进行医学图像分割。
- **2run_medsam_anomaly.py**：运行微调后的 SAM 模型进行异常区域分割。

### 使用方法
1. **依赖安装**：
   确保安装了以下依赖：
   - `torch`
   - `torchvision`
   - `numpy`
   - `opencv-python`
   - `matplotlib`
   - `segment-anything`

2. **路径配置**：
   - `image_dataset/分割/原图`：包含原始图像的文件夹。
   - `image_dataset/分割/标注分割`：包含标注图像的文件夹。

3. **生成掩码**：
   ```bash
   python 2polygon2mask.py
   ```

4. **微调 SAM 模型**：
   ```bash
   python 2finetune_medsam.py
   ```

5. **运行分割模型**：
   ```bash
   python 2run_medsam_anomaly.py
   ```

### 输出
- 生成的掩码保存在 `掩码分割` 文件夹中。
- 分割模型保存在 `logs/best_model_final.pth`。
- 分割结果保存在 `results` 文件夹中。

## 模块 3：分级分期模块

### 功能
- 使用 EfficientNet 模型对医学图像进行分级，判断异常的严重程度。
- 支持 K 折交叉验证和早停机制。
- 提供详细的训练和验证指标可视化。

### 使用方法
1. **依赖安装**：
   确保安装了以下依赖：
   - `torch`
   - `torchvision`
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `tqdm`
   - `Pillow`
   - `matplotlib`

2. **路径配置**：
   - `image_dataset/分期分级/测试集-肿瘤分子分型标注.xlsx`：包含图像分级信息的 Excel 文件。
   - `image_dataset/用于训练的异常图片`：包含异常图像的文件夹。

3. **运行脚本**：
   ```bash
   python 3grading.py
   ```
   4. **测试模型**：
   ```bash
   python 3test.py
   ```

### 输出
- 分级模型保存在 `grading/best_model_final.pth`。
- K 折交叉验证结果保存在 `grading/kfold_results.csv`。
- 各个标签训练过程的可视化图表保存在 `grading/` 文件夹中。

## 项目结构
```
project/
├── 1classify_multiproceed.py
├── 1test.py
├── 2polygon2mask.py
├── 2finetune_medsam.py
├── 2run_medsam_anomaly.py
├── 3grading.py
├── 3test.py
├── image_dataset/
│   ├── 用于训练的所有图片/
│   ├── 用于训练的异常图片/
│   ├── 分割/
│   │   ├── 原图/
│   │   ├── 标注分割/
│   │   ├── 掩码分割/
│   ├── 分期分级/
│   │   ├── 测试集-肿瘤分子分型标注.xlsx
├── classify/
│   ├── best_model_final.pth
│   ├── kfold_results.csv
│   ├── fold_1_training_curves.png
│   ├── fold_1_confusion_matrix.png
│   └── ...
├── logs/
│   ├── best_model_final.pth
├── results/
│   ├── 分割结果.png
├── grading/
│   ├── Grade/
│   ├── HER-2/
│   └── ...
```

---




