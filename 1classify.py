import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import os
import shutil
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import warnings
import sys
import imagehash
import pickle

# 忽略警告信息
warnings.filterwarnings("ignore")

# 设置环境变量限制线程数
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


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
    # 限制PyTorch线程数
    torch.set_num_threads(1)


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


# 自定义数据集类（增加异常处理）
def calculate_phash(file_path):
    """计算图片的感知哈希值"""
    with Image.open(file_path) as img:
        return imagehash.phash(img)

def hamming_distance(hash1, hash2):
    """计算两个哈希值之间的汉明距离"""
    return bin(hash1 ^ hash2).count('1')

class MedicalDataset(Dataset):
    def __init__(self, all_dir, anomaly_dir, transform=None, similarity_threshold=5, cache_dir='cache_train'): #一般图片都是完全一致，阈值设置小一点应该没问题
        self.transform = transform
        self.samples = []
        self.similarity_threshold = similarity_threshold
        self.cache_dir = cache_dir

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 加载或计算all文件夹中所有图片的感知哈希值
        all_hashes_cache_path = os.path.join(cache_dir, 'all_hashes.pkl')
        if os.path.exists(all_hashes_cache_path):
            with open(all_hashes_cache_path, 'rb') as f:
                all_hashes = pickle.load(f)
        else:
            all_hashes = {}
            for root, _, files in os.walk(all_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        try:
                            file_hash = calculate_phash(file_path)
                            all_hashes[file_path] = file_hash
                        except (IOError, OSError, Image.DecompressionBombError) as e:
                            print(f"跳过损坏文件: {file_path}, 错误: {str(e)}")
            with open(all_hashes_cache_path, 'wb') as f:
                pickle.dump(all_hashes, f)

        # 加载或计算anomaly文件夹中所有图片的感知哈希值
        anomaly_hashes_cache_path = os.path.join(cache_dir, 'anomaly_hashes.pkl')
        if os.path.exists(anomaly_hashes_cache_path):
            with open(anomaly_hashes_cache_path, 'rb') as f:
                anomaly_hashes = pickle.load(f)
        else:
            anomaly_hashes = []
            for root, _, files in os.walk(anomaly_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        try:
                            file_hash = calculate_phash(file_path)
                            anomaly_hashes.append(file_hash)
                        except (IOError, OSError, Image.DecompressionBombError) as e:
                            print(f"跳过损坏文件: {file_path}, 错误: {str(e)}")
            with open(anomaly_hashes_cache_path, 'wb') as f:
                pickle.dump(anomaly_hashes, f)

        # 遍历all文件夹中的图片并标记
        for file_path, file_hash in all_hashes.items():
            label = 0  # 默认为正常样本
            for anomaly_hash in anomaly_hashes:
                if hamming_distance(file_hash, anomaly_hash) <= self.similarity_threshold:
                    label = 1  # 如果与任意异常图片足够相似，则标记为异常
                    break
            self.samples.append((file_path, label))

        # 统计类别分布
        labels = [s[1] for s in self.samples]
        self.class_counts = np.bincount(labels)
        print(f"数据集分布 | 正常: {self.class_counts[0]} 异常: {self.class_counts[1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"加载失败: {img_path}, 使用黑色图像替换")
            return torch.zeros(3, 224, 224), label  # 返回空白图像作为占位



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
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pt'):
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
            print(f'验证F1增加 ({self.val_f1_max: .6f} --> {val_f1: .6f}). 保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_f1_max = val_f1


def ensure_empty_directory(dir_path):
    """
    确保指定路径是一个空目录
    - 如果目录不存在，则创建
    - 如果目录存在，则清空目录内容
    """
    # 如果目录不存在，则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录 {dir_path} 已创建")
        return

    # 如果路径存在但不是目录，则抛出异常
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"{dir_path} 不是目录")

    # # 清空目录
    # for filename in os.listdir(dir_path):
    #     file_path = os.path.join(dir_path, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print(f"删除 {file_path} 失败. 原因: {e}")
    #
    # print(f"目录 {dir_path} 已清空")


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
    """
    训练并评估单个折叠的模型
    返回最佳验证指标及模型权重
    """
    best_val_f1 = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        # 添加进度条
        train_bar = tqdm(train_loader, desc=f'Fold {fold_idx + 1} - Epoch {epoch + 1}/{num_epochs} [Train]', file=sys.stdout, dynamic_ncols=True)
        for i, (images, labels) in enumerate(train_bar):
            # print(f'\r{i} / {len(train_loader)}', end='')
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 收集预测结果
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            running_loss += loss.item() * images.size(0)
            # train_bar.set_postfix(loss=loss.item())

        # 计算训练指标
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(all_train_labels, all_train_preds)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        val_bar = tqdm(val_loader, desc=f'Fold {fold_idx + 1} - Epoch {epoch + 1}/{num_epochs} [Val]', file=sys.stdout, dynamic_ncols=True)
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                val_loss += loss.item() * images.size(0)
                # val_bar.set_postfix(loss=loss.item())

        # 计算验证指标
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

        # 保存当前折叠的最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_weights = model.state_dict().copy()
            torch.save(model.state_dict(), f"classify/best_model_fold_{fold_idx + 1}.pth")
            print(f"↻ 保存第{fold_idx + 1}折的新最佳模型")

        # 打印分类报告
        print("\n分类报告（验证集）：")
        print(classification_report(
            all_val_labels, all_val_preds,
            target_names=['正常', '异常'],
            digits=4
        ))

        # 早停检查
        if early_stopping:
            early_stopping(val_metrics['f1'], model)
            if early_stopping.early_stop:
                print("早停触发，停止当前折叠的训练！")
                break

    return {
        'best_f1': best_val_f1,
        'final_val_metrics': val_metrics,
        'best_model_weights': best_model_weights
    }


def main(all_images_dir=None, anomaly_dir=None):
    """
    主函数，允许指定输入目录路径
    """
    print("开始训练二分类模型（单进程版本）")
    set_seed(42)

    # 如果没有指定路径，使用默认路径
    if all_images_dir is None:
        all_images_dir = "image_dataset/用于训练的所有图片"
    if anomaly_dir is None:
        anomaly_dir = "image_dataset/用于训练的异常图片"

    print(f"所有图片文件夹: {all_images_dir}")
    print(f"异常图片文件夹: {anomaly_dir}")

    # 创建完整数据集
    full_dataset = MedicalDataset(all_images_dir, anomaly_dir)

    # K折交叉验证设置
    k_folds = 5  # 5折
    batch_size = 64
    num_epochs = 100

    # 检查GPU可用性并禁用CUDA并行功能
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.enabled = False  # 禁用CUDA并行功能
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU训练")

    # 准备交叉验证
    labels = np.array([label for _, label in full_dataset.samples])
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 存储每折结果
    fold_results = []
    ensure_empty_directory('classify')

    # 开始K折交叉验证
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'=' * 50}")
        print(f"开始第 {fold_idx + 1}/{k_folds} 折训练")
        print(f"{'=' * 50}")

        # 为当前折创建数据集
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
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # 为每个折创建新模型
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, 2, bias=True)
        model = model.to(device)

        # 损失函数与优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        # 早停策略 - 为每个折设置不同的保存路径
        early_stopping = EarlyStopping(
            patience=5,
            verbose=True,
            delta=0.0001,
            path=f'classify/checkpoint_fold_{fold_idx + 1}.pt'
        )

        # 训练并评估当前折
        try:
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

            # 保存此折的结果
            fold_results.append(fold_result)
        except Exception as e:
            print(f"训练第 {fold_idx + 1} 折时出错: {str(e)}")
            continue

        # 释放GPU内存
        del model, train_loader, val_loader, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 检查是否有任何折训练成功
    if not fold_results:
        print("错误: 所有折训练均失败!")
        return

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
            avg_metrics[key] += metrics[key] / len(fold_results)

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

    # 创建最终的汇总模型（可选择使用最佳折叠的权重）
    # final_model = efficientnet_v2_s(weights=None)
    # final_model.classifier[1] = nn.Linear(1280, 2, bias=True)
    # final_model.load_state_dict(fold_results[best_fold_idx]['best_model_weights'])
    # torch.save(final_model.state_dict(), "classify/best_model_final.pth")
    # print("已将最佳折叠的模型保存为最终模型: best_model_final.pth")
    shutil.copy(f"classify/best_model_fold_{best_fold_idx + 1}.pth", "classify/best_model_final.pth")

    # 将结果保存到CSV
    results_df = pd.DataFrame({
        'fold': list(range(1, len(fold_results) + 1)),
        'best_f1': [res['best_f1'] for res in fold_results],
        'accuracy': [res['final_val_metrics']['accuracy'] for res in fold_results],
        'precision': [res['final_val_metrics']['precision'] for res in fold_results],
        'recall': [res['final_val_metrics']['recall'] for res in fold_results],
        'f1': [res['final_val_metrics']['f1'] for res in fold_results]
    })

    # 添加平均行
    avg_row = pd.DataFrame({
        'fold': ['平均'],
        'best_f1': [sum(results_df['best_f1']) / len(fold_results)],
        'accuracy': [avg_metrics['accuracy']],
        'precision': [avg_metrics['precision']],
        'recall': [avg_metrics['recall']],
        'f1': [avg_metrics['f1']]
    })

    results_df = pd.concat([results_df, avg_row])
    results_df.to_csv('classify/kfold_results.csv', index=False)
    print("已将K折交叉验证结果保存到 kfold_results.csv")
    print("\n训练完成!")


if __name__ == '__main__':
    main()
