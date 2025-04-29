import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import shutil
import multiprocessing
import chardet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score



# 设置随机种子确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
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




class GradingDataset(Dataset):
    def __init__(self, excel_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_excel(excel_file)
        self.data['EXAM_DATE'] = self.data['EXAM_DATE'].astype(str)  # 确保 EXAM_DATE 是字符串
        self.data['分级'] = self.data['分级'].map({'低级别': 0, '高级别': 1})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_name = row['PATIENT_NAME']
        exam_date = row['EXAM_DATE']
        file_name = row['FILE_NAME']

        # 确保 exam_date 的格式正确
        if len(exam_date) < 8:
            raise ValueError(f"exam_date should have at least 8 characters, but got {exam_date}")

        # 去掉前导零
        year = exam_date[:4]
        month = str(int(exam_date[4:6]))  # 去掉前导零
        day = str(int(exam_date[6:8]))   # 去掉前导零

        # 构建日期路径
        date_path = os.path.join(self.image_dir, year, month, day).replace("\\", "/")

        # 检查路径是否存在
        if not os.path.exists(date_path):
            raise FileNotFoundError(f"Path does not exist: {date_path}")

        # 在日期路径下搜索包含 PATIENT_NAME 的文件夹
        folder_name = None
        for folder in os.listdir(date_path):
            if patient_name in folder:
                folder_name = folder
                break

        if folder_name is None:
            raise FileNotFoundError(f"No folder found for patient {patient_name} in date path {date_path}")

        # 构建完整的图像路径
        image_path = os.path.join(date_path, folder_name, file_name).replace("\\", "/")


        # 加载图像
        image = Image.open(image_path).convert('RGB')
        label = row['分级']

        if self.transform:
            image = self.transform(image)

        return image, label

# 自定义Subset类应用不同变换
class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# 早停类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = 0
        self.delta = delta
        self.path = path

    def __call__(self, val_f1, model):
        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        if self.verbose:
            print(f'Validation F1 increased ({self.val_f1_max:.6f} --> {val_f1:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_f1_max = val_f1


# 确保目录为空
def ensure_empty_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created.")
        return

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"{dir_path} is not a directory.")

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    print(f"Directory {dir_path} has been cleared.")


# 数据增强配置
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, fold_idx, early_stopping=None):
    
    history = {
    'train_loss': [],
    'val_loss': [],
    'train_f1': [],
    'val_f1': [],
    'lr_history': []  # 记录学习率变化
    }

    best_val_f1 = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        train_bar = tqdm(train_loader, desc=f'Fold {fold_idx + 1} - Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(all_train_labels, all_train_preds)

        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        val_bar = tqdm(val_loader, desc=f'Fold {fold_idx + 1} - Epoch {epoch + 1}/{num_epochs} [Val]')
        all_val_scores = []
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_val_scores.extend(scores)

                val_loss += loss.item() * images.size(0)
                val_bar.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = calculate_metrics(all_val_labels, all_val_preds)

        if scheduler:
            scheduler.step(val_metrics['f1'])
            print(f'学习率: {scheduler.get_last_lr()[0]}')

        # 打印详细指标
        print(f"\nFold {fold_idx + 1} - Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"Precision: {train_metrics['precision']:.4f} | "
              f"Recall: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")

        print(f"Val Loss: {epoch_val_loss:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr_history'].append(scheduler.get_last_lr()[0])

        # 保存当前折叠的最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_weights = model.state_dict().copy()
            torch.save(model.state_dict(), f"grading/best_model_fold_{fold_idx + 1}.pth")
            print(f"↻ 保存第{fold_idx + 1}折的新最佳模型")

            # 保存最佳 epoch 的验证集结果
            _plot_confusion_matrix(all_val_labels, all_val_preds, fold_idx)
            _plot_roc_curve(all_val_labels, all_val_scores, fold_idx)
            _plot_pr_curve(all_val_labels, all_val_scores, fold_idx)  # 新增此行

        # 打印分类报告
        print("\n分类报告（验证集）：")
        print(classification_report(
            all_val_labels, all_val_preds,
            target_names=['低级', '高级'],
            digits=4
        ))

        _plot_training_curves(history, fold_idx)

        # 早停检查
        if early_stopping:
            early_stopping(val_metrics['f1'], model)
            if early_stopping.early_stop:
                print("早停触发，停止当前折叠的训练！")
                break

    return {
        'best_f1': best_val_f1,
        'final_val_metrics': val_metrics,
        'best_model_weights': best_model_weights,
        'history': history
    }

def _plot_training_curves(history, fold_idx):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', linewidth=2, color='tab:blue')
    plt.plot(history['val_loss'], label='Validation', linewidth=2, color='tab:orange')
    plt.title(f'Fold {fold_idx+1} - Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    # F1分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train', linewidth=2, color='tab:green')
    plt.plot(history['val_f1'], label='Validation', linewidth=2, color='tab:red')
    plt.title(f'Fold {fold_idx+1} - F1 Score', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'grading/fold_{fold_idx+1}_training_curves.png', dpi=300)
    plt.close()

def _plot_confusion_matrix(y_true, y_pred, fold_idx):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Fold {fold_idx+1} - Confusion Matrix', fontsize=14)
    plt.colorbar()
    
    # 添加文本标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)
    
    plt.xticks([0, 1], ['Low', 'High'], fontsize=12)
    plt.yticks([0, 1], ['Low', 'High'], fontsize=12)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.savefig(f'grading/fold_{fold_idx+1}_confusion_matrix.png', dpi=300)
    plt.close()

def _plot_roc_curve(y_true, y_scores, fold_idx):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='tab:purple', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='tab:gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Fold {fold_idx+1} - ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(f'grading/fold_{fold_idx+1}_roc_curve.png', dpi=300)
    plt.close()
    

def _plot_pr_curve(y_true, y_scores, fold_idx):
    """绘制PR曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    
    plt.figure()
    plt.plot(recall, precision, color='tab:cyan', lw=2, 
             label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Fold {fold_idx+1} - PR Curve', fontsize=14)
    plt.legend(loc="lower left", fontsize=12)
    plt.savefig(f'grading/fold_{fold_idx+1}_pr_curve.png', dpi=300)
    plt.close()

def main():
    set_seed(42)
    excel_file = "image_dataset/分期分级/分级部分数据.xlsx"
    image_dir = "image_dataset/用于训练的异常图片"

    full_dataset = GradingDataset(excel_file, image_dir)

    k_folds = 4
    batch_size = 16
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
   
    labels = full_dataset.data['分级'].values
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    ensure_empty_directory('grading')

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'=' * 50}")
        print(f"Starting fold {fold_idx + 1}/{k_folds}")
        print(f"{'=' * 50}")

        train_subset = TransformedSubset(
            Subset(full_dataset, train_idx),
            train_transform
        )
        val_subset = TransformedSubset(
            Subset(full_dataset, val_idx),
            val_transform
        )

        # 创建加权采样器
        train_fold_labels = [labels[i] for i in train_idx]
        class_counts = np.bincount(train_fold_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[train_fold_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,  # Windows下建议设为0
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, 2, bias=True)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        early_stopping = EarlyStopping(
            patience=10,
            verbose=True,
            delta=0.0001,
            path=f'grading/checkpoint_fold_{fold_idx + 1}.pt'
        )

        fold_result = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            fold_idx=fold_idx,
            early_stopping=early_stopping
        )

        fold_results.append(fold_result)

        del model, train_loader, val_loader, optimizer, scheduler
        torch.cuda.empty_cache()

    # 汇总所有折的结果
    print("\n" + "=" * 70)
    print("K折交叉验证结果汇总")
    print("=" * 70)

    avg_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }

    best_fold_idx = -1
    best_fold_f1 = -1

    # 计算平均指标并找出最佳模型
    for i, result in enumerate(fold_results):
        metrics = result['final_val_metrics']
        for key in avg_metrics:
            avg_metrics[key] += metrics[key] / k_folds

        print(f"第 {i + 1} 折 - 最佳F1: {result['best_f1']:.4f}")
        if result['best_f1'] > best_fold_f1:
            best_fold_f1 = result['best_f1']
            best_fold_idx = i

    print("\n平均验证指标:")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1 Score: {avg_metrics['f1']:.4f}")

    print(f"\n最佳模型来自第 {best_fold_idx + 1} 折，F1分数: {best_fold_f1:.4f}")
    print(f"最佳模型已保存为 best_model_fold_{best_fold_idx + 1}.pth")

    shutil.copy(f"grading/best_model_fold_{best_fold_idx + 1}.pth", "grading/best_model_final.pth")
    print("已将最佳折叠的模型保存为最终模型: best_model_final.pth")

    # 将结果保存到CSV（可选）
    results_df = pd.DataFrame({
        'fold': list(range(1, k_folds + 1)),
        'best_f1': [res['best_f1'] for res in fold_results],
        'accuracy': [res['final_val_metrics']['accuracy'] for res in fold_results],
        'precision': [res['final_val_metrics']['precision'] for res in fold_results],
        'recall': [res['final_val_metrics']['recall'] for res in fold_results],
        'f1': [res['final_val_metrics']['f1'] for res in fold_results]
    })

    # 添加平均行
    avg_row = pd.DataFrame({
        'fold': ['平均'],
        'best_f1': [sum(results_df['best_f1']) / k_folds],
        'accuracy': [avg_metrics['accuracy']],
        'precision': [avg_metrics['precision']],
        'recall': [avg_metrics['recall']],
        'f1': [avg_metrics['f1']]
    })

    results_df = pd.concat([results_df, avg_row])
    results_df.to_csv('grading/kfold_results.csv', index=False)
    print("已将K折交叉验证结果保存到 kfold_results.csv")

    # 新增全局可视化
    plt.figure()
    for i, res in enumerate(fold_results):
        plt.plot(res['history']['val_f1'], label=f'Fold {i+1}', color=f'C{i}')
    plt.title('Validation Set F1 Scores for All Folds', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig('grading/all_folds_f1.png', dpi=300)
    plt.close()

    # 学习率变化曲线
    plt.figure()
    for i, res in enumerate(fold_results):
        plt.plot(res['history']['lr_history'], label=f'Fold {i+1}', color=f'C{i}')
    plt.title('Learning Rate Curves for All Folds', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.savefig('grading/lr_curves.png', dpi=300)
    plt.close()
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
