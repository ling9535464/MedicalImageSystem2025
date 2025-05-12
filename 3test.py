import os
import torch
import pandas as pd
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

# 设置随机种子确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义评估指标计算函数
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 定义数据集类
class TestDataset(Dataset):
    def __init__(self, data, image_dir, label_column, transform=None):
        self.image_dir = Path(image_dir)
        self.data = data
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 拼接完整路径
        image_path = self.image_dir / row['图片路径']
        
        # 检查路径是否存在
        if not image_path.exists():
            raise FileNotFoundError(f"文件未找到: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 获取真实标签
        label = row[self.label_column]
        return image, label

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, label, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({label})', fontsize=14)
    plt.colorbar()

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.xticks(range(len(set(y_true))), set(y_true), fontsize=12)
    plt.yticks(range(len(set(y_true))), set(y_true), fontsize=12)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.savefig(os.path.join(output_dir, f'{label}_confusion_matrix.png'), dpi=300)
    plt.close()

# 绘制ROC曲线
def plot_roc_curve(y_true, y_scores, label, output_dir):
    num_classes = len(set(y_true))
    y_true_onehot = np.eye(num_classes)[y_true]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='tab:gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve ({label})', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(os.path.join(output_dir, f'{label}_roc_curve.png'), dpi=300)
    plt.close()

# 绘制PR曲线
def plot_pr_curve(y_true, y_scores, label, output_dir):
    num_classes = len(set(y_true))
    y_true_onehot = np.eye(num_classes)[y_true]

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_onehot[:, i], y_scores[:, i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AP = {average_precision[i]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'PR Curve ({label})', fontsize=14)
    plt.legend(loc="lower left", fontsize=12)
    plt.savefig(os.path.join(output_dir, f'{label}_pr_curve.png'), dpi=300)
    plt.close()

# 加载模型
def load_model(label, device, label_class_counts):
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(1280, label_class_counts[label], bias=True)
    model.load_state_dict(torch.load(os.path.join("grading", label, "best_model_final.pth"), map_location=device))
    model.eval()
    return model

# 测试函数
def test_model(label, model, test_loader, device):
    model.to(device)
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {label}"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            scores = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)

    return all_preds, all_labels, all_scores

def main():
    set_seed(42)
    excel_file = "image_dataset\分期分级\验证集2-肿瘤分子分型标注.xlsx"
    image_dir = "."
    full_data = pd.read_excel(excel_file)
    full_data.columns = full_data.columns.str.strip()

    label_columns = [
        'Grade', 'Ki-67_30cutoff', 'Ki-67_40cutoff', 'P53-IHC', 'P53-MUT', 'HER-2',
        'PD-L1(Pemb)', 'PD-L1(Nivo)', 'PD-L1(Atezo)', 'Nectin-4', 'TROP2'
    ]

    label_class_counts = {
        'Grade': 2,
        'Ki-67_30cutoff': 2,
        'Ki-67_40cutoff': 2,
        'P53-IHC': 4,
        'P53-MUT': 2,
        'HER-2': 4,
        'PD-L1(Pemb)': 2,
        'PD-L1(Nivo)': 2,
        'PD-L1(Atezo)': 2,
        'Nectin-4': 3,
        'TROP2': 3
    }

    # 对每个标签进行特殊处理
    for label in label_columns:
        print(f"\n{'=' * 50}")
        print(f"开始处理标签: {label}")
        print(f"{'=' * 50}")

        # 筛选出当前标签非空的数据
        label_data = full_data.dropna(subset=[label]).reset_index(drop=True)

        # 特殊处理逻辑
        # if label == 'P53-MUT':
        #     # 将 'Y' 映射为 1，'N' 映射为 0
        #     label_data[label] = label_data[label].map({'Y': 1, 'N': 0}).astype(int)
        #elif label in ['Nectin-4', 'TROP2']:
        if label == 'TROP2': #对验证集2
            # 将标签减去 1
            label_data[label] = label_data[label].astype(int) - 1
        else:
            # 其他标签直接转换为整数类型
            label_data[label] = label_data[label].astype(int)

        # 创建测试数据集
        test_dataset = TestDataset(label_data, image_dir, label_column=label, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        model = load_model(label, device, label_class_counts)

        # 测试模型
        y_pred, y_true, y_scores = test_model(label, model, test_loader, device)

        # 计算评估指标
        metrics = calculate_metrics(y_true, y_pred)
        print(f"\n{label} - 测试结果")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

        # 可视化结果
        output_dir = f"grading/{label}/test2_results"
        os.makedirs(output_dir, exist_ok=True)

        plot_confusion_matrix(y_true, y_pred, label, output_dir)
        plot_roc_curve(y_true, np.array(y_scores), label, output_dir)
        plot_pr_curve(y_true, np.array(y_scores), label, output_dir)
   

if __name__ == '__main__':
    main()