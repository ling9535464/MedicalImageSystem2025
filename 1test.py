import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import os
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc, roc_curve, roc_auc_score
from tqdm import tqdm
import pandas as pd
import imagehash
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  # <<< 新增导入


def calculate_phash(file_path):
    """计算图片的感知哈希值"""
    with Image.open(file_path) as img:
        return imagehash.phash(img)

def hamming_distance(hash1, hash2):
    return hash1 - hash2

# 测试集数据加载类
class TestDataset(Dataset): #由于测试集的文件名混乱，无法使用训练集同款标签设置，故使用汉明距离代替
    def __init__(self, all_dir, anomaly_dir, transform=None, similarity_threshold=0, cache_dir='cache_test'): #一般图片都是完全一致，阈值设置小一点应该没问题
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

        # 打印数据集信息
        if self.samples:
            labels = [s[1] for s in self.samples]
            class_counts = np.bincount(labels, minlength=2)
            print(f"测试集图片总数: {len(self.samples)}")
            print(f"类别分布 | 正常: {class_counts[0]} 异常: {class_counts[1]}")
        else:
            print("警告: 未找到有效的测试图片!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path  # 返回图片路径方便后续分析
        except Exception as e:
            print(f"加载失败: {img_path}, 使用黑色图像替换")
            return torch.zeros(3, 224, 224), label, img_path


def evaluate_model(model_path, test_dir, anomaly_dir=None, batch_size=32):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280, 2, bias=True)
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    print(f"模型已加载: {model_path}")

    # 数据预处理变换，与验证集使用相同的变换
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建测试数据集和数据加载器
    test_dataset = TestDataset(test_dir, anomaly_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 评估模型
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="评估进度"):
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取异常类的概率
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    # 计算整体指标
    print("\n--- 整体评估指标 ---")
    report = classification_report(
        all_labels, all_preds,
        target_names=['正常', '异常'],
        digits=4,
        zero_division=0
    )
    print(report)

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()

    # 添加文本标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.xticks([0, 1], ['Normal', 'Anomaly'], fontsize=12)
    plt.yticks([0, 1], ['Normal', 'Anomaly'], fontsize=12)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.savefig('valset/valset2_confusion_matrix.png', dpi=300)
    plt.close()
    print("混淆矩阵图已保存为 'valset/valset2_confusion_matrix.png'")

    # 计算每个类别的F1分数
    f1_normal = f1_score(
        [1 if label == 0 else 0 for label in all_labels],
        [1 if pred == 0 else 0 for pred in all_preds],
        zero_division=0
    )
    f1_anomaly = f1_score(
        [1 if label == 1 else 0 for label in all_labels],
        [1 if pred == 1 else 0 for pred in all_preds],
        zero_division=0
    )

    print(f"正常图片 F1 分数: {f1_normal:.4f}")
    print(f"异常图片 F1 分数: {f1_anomaly:.4f}")
    
    if sum(all_labels) == 0:
        print("\n警告：测试集无异常样本，无法计算PR AUC")
        pr_auc = float('nan')
    else:
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)
        print(f"\nPR AUC: {pr_auc:.4f}")

    # 绘制Precision-Recall曲线图
    plt.figure()
    plt.plot(recall, precision, color='tab:cyan', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig('valset/valset2_pr_curve.png')
    plt.close()
    print("Precision-Recall曲线图已保存为 'valset/valset2_pr_curve.png'")

    # <<< 新增：ROC曲线 >>> 
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='tab:purple', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='tab:gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rates')
    plt.ylabel('True positive rates')
    plt.title('ROC cureve')
    plt.legend(loc="lower right")
    plt.savefig('valset/valset2_roc_curve.png')
    plt.close()
    print("ROC曲线图已保存为 'valset/valset2_roc_curve.png'")

    

    # # 创建结果详情，包括预测错误的图片
    # results_df = pd.DataFrame({
    #     '图片路径': all_paths,
    #     '真实标签': ['正常' if label == 0 else '异常' for label in all_labels],
    #     '预测标签': ['正常' if pred == 0 else '异常' for pred in all_preds],
    #     '是否正确': [pred == label for pred, label in zip(all_preds, all_labels)]
    # })

    # # 保存错误分类的图片信息
    # error_df = results_df[~results_df['是否正确']]
    # if len(error_df) > 0:
    #     error_df.to_csv('错误分类图片.csv', index=False, encoding='utf-8-sig')
    #     print(f"共有 {len(error_df)} 张图片分类错误，详情已保存至 '错误分类图片.csv'")
    # else:
    #     print("\n所有图片均正确分类！")

    # 创建结果详情，包括所有图片的分类结果
    results_df = pd.DataFrame({
        '图片路径': all_paths,
        '真实标签': ['正常' if label == 0 else '异常' for label in all_labels],
        '预测标签': ['正常' if pred == 0 else '异常' for pred in all_preds],
        '是否正确': [pred == label for pred, label in zip(all_preds, all_labels)]
    })

    # 保存所有图片的分类结果到CSV文件
    results_df.to_csv('valset/valset2_所有图片分类结果.csv', index=False, encoding='utf-8-sig')
    print("所有图片的分类结果已保存至 'valset/valset2_所有图片分类结果.csv'")


    return {
        'f1_overall': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_normal': f1_normal,
        'f1_anomaly': f1_anomaly,
        'pr_auc': pr_auc,
        'results_df': results_df
    }


if __name__ == '__main__':
    # 配置路径
    model_path = "classify/best_model_final.pth"  # 训练好的模型路径
    test_dir = "模拟用户使用的测试图片/所有图片/验证集2-all"  # 测试图片文件夹
    anomaly_dir = "模拟用户使用的测试图片/异常图片/验证集2-tumor"  # 测试集中的异常图片文件夹

    # 评估模型
    results = evaluate_model(model_path, test_dir, anomaly_dir)

    print("\n--- 评估完成 ---")
