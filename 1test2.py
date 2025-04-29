import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil  

# 数据集加载类
class DatasetForPrediction(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []

        # 遍历数据集文件夹中的图片
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    self.samples.append(file_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"加载失败: {img_path}, 使用黑色图像替换")
            return torch.zeros(3, 224, 224), img_path


def predict_with_model(model_path, data_dir, batch_size=32):
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

    # 数据预处理变换
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集和数据加载器
    dataset = DatasetForPrediction(data_dir, transform=data_transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 进行预测
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for images, paths in tqdm(data_loader, desc="预测进度"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_paths.extend(paths)

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '图片路径': all_paths,
        '预测标签': ['正常' if pred == 0 else '异常' for pred in all_preds]
    })

    # 保存结果到CSV文件
    results_df.to_csv('验证集2预测结果.csv', index=False, encoding='utf-8-sig')
    print("预测结果已保存至 '验证集2预测结果.csv'")

    # 将分类为异常的图片复制到一个新的文件夹中
    copy_abnormal_images(results_df, "./abnormal_images_validation2")


def copy_abnormal_images(results_df, target_dir):
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    # 筛选出预测为异常的图片
    abnormal_df = results_df[results_df['预测标签'] == '异常']

    # 复制图片到目标文件夹
    for img_path in abnormal_df['图片路径']:
        try:
            shutil.copy(img_path, target_dir)
        except Exception as e:
            print(f"复制图片失败: {img_path}, 错误: {str(e)}")

    print(f"所有预测为异常的图片已复制到 {target_dir}")


if __name__ == '__main__':
    # 配置路径
    model_path = "classify/best_model_final.pth"  # 训练好的模型路径
    data_dir = "./模拟用户使用的测试图片/验证集2 - 所有图片"  # 需要预测的数据集文件夹

    # 进行预测
    predict_with_model(model_path, data_dir)