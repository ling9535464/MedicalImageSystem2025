import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
import copy
import random
import shutil
from sklearn.model_selection import KFold
import torch.nn.functional as F


class MedicalSegmentationDataset(Dataset):
    """医学图像分割数据集类"""

    def __init__(self, image_dir, mask_dir, transform=None):
        """初始化数据集"""
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 获取所有原始图像文件
        self.image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                           glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                           glob.glob(os.path.join(image_dir, "*.png"))
        self.image_files = sorted(self.image_files)

        # 筛选有对应掩码的图像
        valid_images = []
        self.mask_files = []

        for img_path in self.image_files:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")

            if os.path.exists(mask_path):
                valid_images.append(img_path)
                self.mask_files.append(mask_path)

        self.image_files = valid_images
        print(f"找到 {len(self.image_files)} 对有效的图像-掩码对")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像和掩码
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # 使用OpenCV读取中文路径的图像
        img_data = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_data = np.fromfile(mask_path, dtype=np.uint8)
        mask = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)

        # 确保掩码是二值图像
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # 调整图像大小为1024x1024（SAM模型要求）
        original_h, original_w = image.shape[:2]
        if original_h != 1024 or original_w != 1024:
            image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # 应用数据增强（如果有）
        if self.transform:
            # 将图像和掩码转换为PIL格式以用于变换
            image_pil = Image.fromarray(image)
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)

            # 使用相同的随机种子确保图像和掩码一致变换
            seed = torch.randint(0, 2147483647, (1,)).item()

            torch.manual_seed(seed)
            image_pil = self.transform(image_pil)

            torch.manual_seed(seed)
            mask_pil = self.transform(mask_pil)

            # 转换回numpy
            image = np.array(image_pil)
            mask = np.array(mask_pil)

            # 确保掩码是二值的
            mask = (mask > 127).astype(np.float32)

        # 转换为PyTorch张量
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0  # [3, H, W]
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]

        # 生成图像嵌入所需的提示点
        foreground_points = []
        background_points = []

        binary_mask = mask.squeeze().numpy() > 0.5

        # 找到掩码中的前景像素坐标
        y_indices, x_indices = np.where(binary_mask)
        if len(y_indices) > 0:
            # 从前景中随机选择点
            idx_choices = np.random.choice(len(y_indices), size=min(3, len(y_indices)), replace=False)
            for idx in idx_choices:
                foreground_points.append((x_indices[idx], y_indices[idx]))

        # 找到掩码中的背景像素坐标
        y_indices, x_indices = np.where(~binary_mask)
        if len(y_indices) > 0:
            # 从背景中随机选择点
            idx_choices = np.random.choice(len(y_indices), size=min(3, len(y_indices)), replace=False)
            for idx in idx_choices:
                background_points.append((x_indices[idx], y_indices[idx]))

        # 将前景点和背景点合并为提示点
        point_coords = []
        point_labels = []

        for point in foreground_points:
            point_coords.append(point)
            point_labels.append(1)  # 1表示前景

        for point in background_points:
            point_coords.append(point)
            point_labels.append(0)  # 0表示背景

        # 转换为PyTorch张量
        if point_coords:
            point_coords = torch.tensor(point_coords, dtype=torch.float)
            point_labels = torch.tensor(point_labels, dtype=torch.int)
        else:
            # 如果没有点，创建空张量
            point_coords = torch.zeros((0, 2), dtype=torch.float)
            point_labels = torch.zeros(0, dtype=torch.int)

        sample = {
            'image': image,
            'mask': mask,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'image_path': img_path
        }

        return sample


class DiceLoss(nn.Module):
    """Dice损失函数，处理不同分辨率的掩码"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # 检查并调整预测掩码的大小以匹配目标掩码
        if predictions.shape != targets.shape:
            predictions = F.interpolate(
                predictions,
                size=targets.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # 应用sigmoid激活
        predictions = torch.sigmoid(predictions)

        # 展平预测和目标
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # 计算交集
        intersection = (predictions * targets).sum()

        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # 返回Dice损失
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal损失函数，处理不同分辨率的掩码"""

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 检查并调整预测掩码的大小以匹配目标掩码
        if inputs.shape != targets.shape:
            inputs = F.interpolate(
                inputs,
                size=targets.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # 应用sigmoid激活
        inputs = torch.sigmoid(inputs)

        # 展平输入和目标
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 二元交叉熵
        bce = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='none')

        # 计算概率
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # 计算alpha加权
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 应用gamma权重
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce

        # 根据reduction模式返回结果
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """提前停止训练的类，用于防止过拟合"""

    def __init__(self, patience=5, min_delta=0, checkpoint_path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # 验证损失减小，保存模型并重置计数器
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return True
        else:
            # 验证损失没有减小
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class SamFineTuner:
    """SAM模型微调类"""

    def __init__(self, model_type, checkpoint_path, device='cuda'):
        self.device = device
        self.model_type = model_type

        # 初始化SAM模型
        print(f"正在加载预训练的SAM模型 {model_type} 从 {checkpoint_path}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)

        # 创建一个深拷贝用于微调
        self.finetune_sam = copy.deepcopy(self.sam)

        # 冻结编码器参数
        for param in self.finetune_sam.image_encoder.parameters():
            param.requires_grad = False

        # 解冻掩码解码器参数
        for param in self.finetune_sam.mask_decoder.parameters():
            param.requires_grad = True

        # 解冻提示编码器参数
        for param in self.finetune_sam.prompt_encoder.parameters():
            param.requires_grad = True

    def train_one_fold(self, train_loader, val_loader, fold_idx, log_dir, epochs=10, lr=5e-6, weight_decay=1e-3, patience=5):
        """训练一折交叉验证数据"""
        # 创建检查点目录
        checkpoint_dir = os.path.join(log_dir, f"fold_{fold_idx}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

        # 设置优化器
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.finetune_sam.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # 设置学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

        # 设置损失函数
        dice_loss = DiceLoss()
        focal_loss = FocalLoss()

        # 设置提前停止
        early_stopping = EarlyStopping(patience=patience, checkpoint_path=best_model_path)

        # 跟踪最佳模型
        best_val_loss = float('inf')

        # 训练循环
        for epoch in range(epochs):
            print(f"Fold {fold_idx + 1}, Epoch {epoch + 1}/{epochs}")

            # 训练阶段
            self.finetune_sam.train()
            train_loss = 0.0

            # 使用tqdm显示进度条
            progress_bar = tqdm(train_loader, desc=f"训练", unit="batch")
            for batch_idx, batch in enumerate(progress_bar):
                # 获取数据
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                point_coords = batch['point_coords'].to(self.device)
                point_labels = batch['point_labels'].to(self.device)

                # 获取原始图像尺寸
                batch_size, _, orig_h, orig_w = images.shape

                # 清除梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.no_grad():
                    # 使用图像编码器生成图像嵌入
                    image_embeddings = self.finetune_sam.image_encoder(images)

                # 对每个样本单独处理提示点
                loss_batch = 0

                for i in range(batch_size):
                    # 获取当前样本的提示点
                    curr_points = point_coords[i]
                    curr_labels = point_labels[i]

                    # 如果没有提示点，创建一个随机点
                    if curr_points.shape[0] == 0:
                        curr_points = torch.tensor([[orig_w // 2, orig_h // 2]], dtype=torch.float).to(self.device)
                        curr_labels = torch.tensor([1], dtype=torch.int).to(self.device)

                    # 编码提示点
                    sparse_embeddings, dense_embeddings = self.finetune_sam.prompt_encoder(
                        points=(curr_points.unsqueeze(0), curr_labels.unsqueeze(0)),
                        boxes=None,
                        masks=None,
                    )

                    # 调用掩码解码器
                    mask_predictions, _ = self.finetune_sam.mask_decoder(
                        image_embeddings=image_embeddings[i:i + 1],
                        image_pe=self.finetune_sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    # 计算损失 - 损失函数内部会处理尺寸不匹配的问题
                    curr_mask = masks[i:i + 1]
                    dl = dice_loss(mask_predictions, curr_mask)
                    fl = focal_loss(mask_predictions, curr_mask)
                    loss = (dl + fl) / 2
                    loss_batch += loss

                # 平均批次损失
                loss = loss_batch / batch_size

                # 反向传播和优化
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.finetune_sam.parameters(), max_norm=1.0)

                optimizer.step()

                # 更新进度条
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            # 计算平均训练损失
            train_loss /= len(train_loader)

            # 验证阶段
            val_loss, val_accuracy = self.validate(val_loader)

            # 每个epoch后更新学习率
            scheduler.step(val_loss)
            print(f'学习率: {scheduler.get_last_lr()[0]}')

            print(
                f"Fold {fold_idx + 1}, Epoch {epoch + 1}/{epochs}, "
                f"训练损失: {train_loss:.4f}, "
                f"验证损失: {val_loss:.4f}, "
                f"验证准确率: {val_accuracy:.4f}"
            )

            # 检查是否达到提前停止条件
            early_stopping(val_loss, self.finetune_sam)
            if early_stopping.early_stop:
                print(f"提前停止训练，最佳验证损失: {early_stopping.best_loss:.4f}")
                break

            # 更新最佳验证损失
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # 加载最佳模型
        if os.path.exists(best_model_path):
            self.finetune_sam.load_state_dict(torch.load(best_model_path, weights_only=False))

        return best_val_loss, best_model_path

    def validate(self, val_loader):
        """在验证集上评估模型"""
        # 设置损失函数
        dice_loss = DiceLoss()
        focal_loss = FocalLoss()

        # 将模型设置为评估模式
        self.finetune_sam.eval()

        val_loss = 0.0
        total_correct = 0
        total_pixels = 0

        # 使用tqdm显示进度条
        progress_bar = tqdm(val_loader, desc=f"验证", unit="batch")
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # 获取数据
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                point_coords = batch['point_coords'].to(self.device)
                point_labels = batch['point_labels'].to(self.device)

                # 获取原始图像尺寸
                batch_size, _, orig_h, orig_w = images.shape

                # 前向传播
                image_embeddings = self.finetune_sam.image_encoder(images)

                # 对每个样本单独处理提示点
                loss_batch = 0
                batch_correct = 0
                batch_pixels = 0

                for i in range(batch_size):
                    # 获取当前样本的提示点
                    curr_points = point_coords[i]
                    curr_labels = point_labels[i]

                    # 如果没有提示点，创建一个随机点
                    if curr_points.shape[0] == 0:
                        curr_points = torch.tensor([[orig_w // 2, orig_h // 2]], dtype=torch.float).to(self.device)
                        curr_labels = torch.tensor([1], dtype=torch.int).to(self.device)

                    # 编码提示点
                    sparse_embeddings, dense_embeddings = self.finetune_sam.prompt_encoder(
                        points=(curr_points.unsqueeze(0), curr_labels.unsqueeze(0)),
                        boxes=None,
                        masks=None,
                    )

                    # 调用掩码解码器
                    mask_predictions, _ = self.finetune_sam.mask_decoder(
                        image_embeddings=image_embeddings[i:i + 1],
                        image_pe=self.finetune_sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    # 计算损失
                    curr_mask = masks[i:i + 1]
                    dl = dice_loss(mask_predictions, curr_mask)
                    fl = focal_loss(mask_predictions, curr_mask)
                    loss = (dl + fl) / 2
                    loss_batch += loss

                # 将预测掩码上采样到原始尺寸
                pred_mask = (torch.sigmoid(mask_predictions) > 0.5).float()
                true_mask = F.interpolate(masks, size=pred_mask.shape[-2:], mode='nearest')  # 调整真实掩码尺寸

                correct = (pred_mask == true_mask).sum().item()
                pixels = true_mask.numel()
                batch_correct += correct
                batch_pixels += pixels

                # 平均批次损失
                loss = loss_batch / batch_size

                # 更新总损失
                val_loss += loss.item()
                total_correct += batch_correct
                total_pixels += batch_pixels
                progress_bar.set_postfix({'loss': loss.item()})

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        accuracy = total_correct / total_pixels if total_pixels > 0 else 0
        return avg_val_loss, accuracy

    def train_with_kfold(self, dataset, k=5, epochs=10, batch_size=2, log_dir='./logs', patience=5, lr=5e-6, weight_decay=1e-3):
        """使用K折交叉验证训练模型"""
        # 创建日志目录
        if os.path.exists(log_dir) and not os.path.isdir(log_dir):
            # 如果是文件而不是目录，先删除它
            os.remove(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # 创建KFold对象
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # 记录每折的性能
        fold_results = []

        # 为每一折训练模型
        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset)))):
            print(f"\n============= Fold {fold_idx + 1}/{k} =============")

            # 重置模型，每折都从预训练模型开始
            if fold_idx > 0:
                self.finetune_sam = copy.deepcopy(self.sam)
                # 冻结编码器参数
                for param in self.finetune_sam.image_encoder.parameters():
                    param.requires_grad = False
                # 解冻掩码解码器参数
                for param in self.finetune_sam.mask_decoder.parameters():
                    param.requires_grad = True
                # 解冻提示编码器参数
                for param in self.finetune_sam.prompt_encoder.parameters():
                    param.requires_grad = True
                self.finetune_sam.to(self.device)

            # 创建数据加载器
            train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=0,  # 设为0以避免多进程问题
                pin_memory=True
            )

            val_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=0,  # 设为0以避免多进程问题
                pin_memory=True
            )

            print(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")

            # 训练当前折
            best_val_loss, best_model_path = self.train_one_fold(
                train_loader=train_loader,
                val_loader=val_loader,
                fold_idx=fold_idx,
                log_dir=log_dir,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience
            )

            # 记录结果
            fold_results.append((fold_idx, best_val_loss, best_model_path))

        # 找到性能最好的折
        best_fold = min(fold_results, key=lambda x: x[1])
        print(f"\n最佳模型来自 Fold {best_fold[0] + 1}, 验证损失: {best_fold[1]:.4f}")

        # 加载最佳模型
        best_model_path = best_fold[2]
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)

        # 创建最终模型路径
        final_model_path = os.path.join(log_dir, "best_model_final.pth")
        self.save_model(final_model_path)

        return final_model_path

    def save_model(self, path):
        """保存模型权重"""
        torch.save(self.finetune_sam.state_dict(), path)
        print(f"模型已保存到 {path}")

    def load_model(self, path):
        """加载模型权重"""
        self.finetune_sam.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        print(f"模型已从 {path} 加载")

    def predict(self, image_path, points=None, output_path=None):
        """使用微调后的模型进行预测"""
        # 设置为评估模式
        self.finetune_sam.eval()

        # 读取图像
        img_data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存原始图像尺寸以用于后处理
        original_h, original_w = image.shape[:2]

        # 调整图像大小为1024x1024以符合SAM模型
        if original_h != 1024 or original_w != 1024:
            image_input = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        else:
            image_input = image.copy()

        # 转换为PyTorch张量
        image_tensor = torch.from_numpy(image_input).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)

        # 前向传播
        with torch.no_grad():
            # 获取图像嵌入
            image_embeddings = self.finetune_sam.image_encoder(image_tensor)

            # 如果没有提供点，在图像中心创建一个点
            if points is None:
                h, w = image_input.shape[:2]
                points = [(w // 2, h // 2, 1)]  # 中心点，标签为前景

            # 准备点坐标和标签
            point_coords = []
            point_labels = []
            for x, y, label in points:
                # 如果提供的点是基于原始图像尺寸，需要调整到1024x1024
                if original_h != 1024 or original_w != 1024:
                    x = int(x * (1024 / original_w))
                    y = int(y * (1024 / original_h))
                point_coords.append([x, y])
                point_labels.append(label)

            point_coords = torch.tensor(point_coords, dtype=torch.float).to(self.device)
            point_labels = torch.tensor(point_labels, dtype=torch.int).to(self.device)

            # 编码提示点
            sparse_embeddings, dense_embeddings = self.finetune_sam.prompt_encoder(
                points=(point_coords.unsqueeze(0), point_labels.unsqueeze(0)),
                boxes=None,
                masks=None,
            )

            # 调用掩码解码器
            mask_predictions, _ = self.finetune_sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.finetune_sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # 将掩码上采样到原始图像大小
            mask_predictions = F.interpolate(
                mask_predictions,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )

        # 将预测结果转换为二值掩码
        mask = mask_predictions[0, 0].sigmoid().cpu().numpy() > 0.5
        mask = mask.astype(np.uint8) * 255

        # 将掩码调整回原始图像尺寸
        if original_h != 1024 or original_w != 1024:
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # 可视化和保存结果
        if output_path:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("原始图像")
            plt.axis("off")

            # 绘制提示点
            for (x, y, label) in points:
                color = 'green' if label == 1 else 'red'
                plt.plot(x, y, 'o', color=color, markersize=8)

            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("预测掩码")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            # 创建带有掩码叠加的可视化
            overlay = image.copy()
            mask_colored = np.zeros_like(image)
            mask_colored[mask > 0] = [0, 255, 0]  # 绿色掩码
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            plt.imshow(overlay)
            plt.title("掩码叠加")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

        return mask


def create_data_transforms():
    """创建适合SAM模型的数据增强转换"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])
    return transform


def find_checkpoint_path(base_dir, model_name="sam_vit_b_01ec64.pth"):
    """查找模型检查点路径"""
    # 首先检查当前目录
    if os.path.exists(model_name):
        return model_name

    # 然后检查指定目录
    if os.path.exists(os.path.join(base_dir, model_name)):
        return os.path.join(base_dir, model_name)

    # 最后递归搜索
    for root, dirs, files in os.walk(base_dir):
        if model_name in files:
            return os.path.join(root, model_name)

    return None


def filter_mask_files(mask_dir):
    """过滤出真正的掩码文件，排除比较图片文件"""
    all_files = glob.glob(os.path.join(mask_dir, "*.*"))
    mask_files = [f for f in all_files if "_mask.png" in f]
    return mask_files


def main(image_dir=None, mask_dir=None):
    """主函数"""
    # 设置目录路径
    base_dir = "."
    # 如果没有指定路径，使用默认路径
    if image_dir is None:
        image_dir = "./image_dataset/分割/原图"
    if mask_dir is None:
        mask_dir = "./image_dataset/分割/掩码分割"
    log_dir = "./logs"
    results_dir = "./results"
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号


    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 寻找预训练模型
    model_path = find_checkpoint_path(base_dir, "sam_vit_b_01ec64.pth")
    if model_path is None:
        print("错误: 找不到SAM预训练模型文件 'sam_vit_b_01ec64.pth'")
        return

    print(f"使用预训练模型: {model_path}")

    # 创建数据增强
    transform = create_data_transforms()

    # 创建数据集
    dataset = MedicalSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)

    # 打印掩码文件数量
    mask_files = filter_mask_files(mask_dir)
    print(f"在掩码目录中找到 {len(mask_files)} 个掩码文件")

    # 初始化SAM微调器
    fine_tuner = SamFineTuner(
        model_type="vit_b",
        checkpoint_path=model_path,
        device=device
    )

    # 使用K折交叉验证训练模型
    best_model_path = fine_tuner.train_with_kfold(
        dataset=dataset,
        k=5,  # 5折交叉验证
        epochs=30,
        batch_size=3,  # 减小批量大小以避免内存问题
        log_dir=log_dir,
        patience=5,
        lr=5e-6,
        weight_decay=1e-3
    )

    # 加载最佳模型
    if os.path.exists(best_model_path):
        fine_tuner.load_model(best_model_path)

    # 在测试图像上进行预测和评估
    test_images = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.jpeg")) + glob.glob(os.path.join(image_dir, "*.png"))

    # 随机选择5张图像进行测试
    if len(test_images) > 5:
        test_images = random.sample(test_images, 5)

    for image_path in test_images:
        # 提取图像名称
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]

        # 检查是否有对应的掩码
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")

        # 读取掩码以获取提示点
        if os.path.exists(mask_path):
            mask_data = np.fromfile(mask_path, dtype=np.uint8)
            mask = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)

            # 获取前景点
            y_indices, x_indices = np.where(mask > 127)

            if len(y_indices) > 0:
                # 选择三个前景点
                points = []
                for _ in range(min(3, len(y_indices))):
                    idx = np.random.randint(0, len(y_indices))
                    points.append((x_indices[idx], y_indices[idx], 1))
            else:
                # 如果没有前景，使用图像中心点
                h, w = mask.shape
                points = [(w // 2, h // 2, 1)]

            # 进行预测
            output_path = os.path.join(results_dir, f"{base_name}_prediction.png")
            fine_tuner.predict(image_path, points=points, output_path=output_path)
            print(f"预测结果已保存到 {output_path}")
        else:
            # 如果没有掩码，使用图像中心点
            img_data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            points = [(w // 2, h // 2, 1)]

            # 进行预测
            output_path = os.path.join(results_dir, f"{base_name}_prediction.png")
            fine_tuner.predict(image_path, points=points, output_path=output_path)
            print(f"预测结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
