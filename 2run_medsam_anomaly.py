import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from segment_anything import sam_model_registry
import time
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# =================== 配置参数 ===================
# 微调后的SAM模型路径
MODEL_PATH = "./logs/best_model_final.pth"
# MODEL_PATH = "sam_vit_b_01ec64.pth"
# SAM模型类型
MODEL_TYPE = "vit_b"
# 是否使用交互式选择点 (True/False)
INTERACTIVE_MODE = False
# 输入目录
INPUT_DIR = "./image_dataset/用于训练的异常图片"
# 输出目录
OUTPUT_DIR = "./results"


# ===============================================

class MedicalImageSegmenter:
    """医学图像分割器类，用于加载SAM模型并进行图像分割"""

    def __init__(self, model_path, model_type="vit_b"):
        """
        初始化分割器

        参数:
            model_path (str): 微调后的SAM模型路径
            model_type (str): SAM模型类型，默认为"vit_b"
        """
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载SAM模型
        print(f"正在加载SAM模型: {model_type}，从路径: {model_path}")
        self.model = sam_model_registry[model_type]()

        # 使用微调后的权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print("模型已加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请确保模型路径正确，并已完成SAM模型的微调")
            self.model = None

    def preprocess_image(self, image_path):
        """
        预处理图像

        参数:
            image_path (str): 图像路径

        返回:
            tuple: 原始图像和预处理后的张量
        """
        # 读取图像（支持中文路径）
        img_data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存原始图像尺寸
        original_h, original_w = image.shape[:2]

        # 调整图像大小为1024x1024以符合SAM模型
        if original_h != 1024 or original_w != 1024:
            image_input = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        else:
            image_input = image.copy()

        # 转换为PyTorch张量
        image_tensor = torch.from_numpy(image_input).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        return image, image_tensor

    def segment_image(self, image_tensor, points=None):
        """
        分割图像

        参数:
            image_tensor (torch.Tensor): 预处理后的图像张量
            points (list): 提示点列表，每个点是一个形如(x,y,label)的元组，label为0表示背景，1表示前景

        返回:
            numpy.ndarray: 二值掩码
        """
        if self.model is None:
            print("错误: 模型未成功加载")
            return np.zeros((1024, 1024), dtype=np.uint8)

        # 将张量移至正确的设备
        image_tensor = image_tensor.to(self.device)

        # 获取图像嵌入
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(image_tensor)

            # 如果没有提供点，在图像中心创建一个点
            if points is None or len(points) == 0:
                h, w = image_tensor.shape[2:]
                points = [(w // 2, h // 2, 1)]  # 中心点，标签为前景

            # 准备点坐标和标签
            point_coords = []
            point_labels = []
            for x, y, label in points:
                point_coords.append([x, y])
                point_labels.append(label)

            point_coords = torch.tensor(point_coords, dtype=torch.float).to(self.device)
            point_labels = torch.tensor(point_labels, dtype=torch.int).to(self.device)

            # 编码提示点
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(point_coords.unsqueeze(0), point_labels.unsqueeze(0)),
                boxes=None,
                masks=None,
            )

            # 调用掩码解码器
            mask_predictions, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
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

        return mask

    # def visualize_results(self, image, mask, output_path=None, points=None): #对单张图片的可视化
    #     """
    #     可视化分割结果
    #
    #     参数:
    #         image (numpy.ndarray): 原始图像
    #         mask (numpy.ndarray): 预测掩码
    #         output_path (str): 输出路径
    #         points (list): 提示点列表
    #     """
    #     # 调整掩码大小以匹配原始图像
    #     if image.shape[:2] != mask.shape:
    #         mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    #
    #     # 创建图像
    #     plt.figure(figsize=(15, 5))
    #
    #     # 原始图像
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image)
    #     plt.title("原始图像")
    #     plt.axis("off")
    #
    #     # 如果提供了点，绘制它们
    #     if points:
    #         for x, y, label in points:
    #             color = 'green' if label == 1 else 'red'
    #             plt.plot(x, y, 'o', color=color, markersize=10)
    #
    #     # 预测掩码
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(mask, cmap='gray')
    #     plt.title("预测掩码")
    #     plt.axis("off")
    #
    #     # 叠加图像
    #     plt.subplot(1, 3, 3)
    #     overlay = image.copy()
    #     mask_colored = np.zeros_like(image)
    #     mask_colored[mask > 0] = [0, 255, 0]  # 绿色掩码
    #     overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    #     plt.imshow(overlay)
    #     plt.title("掩码叠加")
    #     plt.axis("off")
    #
    #     plt.tight_layout()
    #
    #     # 保存图像
    #     if output_path:
    #         output_dir = os.path.dirname(output_path)
    #         if output_dir and not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #         plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #         print(f"分割结果已保存到: {output_path}")
    #
    #     # 显示图像
    #     plt.show()

    def visualize_results(self, image, mask, output_path=None, points=None):
        """
        可视化分割结果
        """
        # 调整掩码大小以匹配原始图像
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 如果图像过大，调整图像大小
        if image.shape[0] > 1024 or image.shape[1] > 1024:
            image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # 创建图像
        plt.figure(figsize=(10, 8))  # 减小图像尺寸

        # 原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("原始图像", fontsize=16)
        plt.axis("off")

        # 如果提供了点，绘制它们
        if points:
            for x, y, label in points:
                color = 'green' if label == 1 else 'red'
                plt.plot(x, y, 'o', color=color, markersize=12)

        # 叠加图像
        plt.subplot(1, 2, 2)
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 0]  # 绿色掩码
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        plt.imshow(overlay)
        plt.title("异常区域分割结果", fontsize=16)
        plt.axis("off")

        plt.tight_layout(pad=3.0)

        # 保存图像
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')  # 降低 DPI
            print(f"分割结果已保存到: {output_path}")

        # 显示图像
        # plt.show()

    def process_image(self, image_path, output_path=None, points=None):
        """
        处理单张图像的完整流程

        参数:
            image_path (str): 输入图像路径
            output_path (str): 输出路径
            points (list): 提示点列表

        返回:
            tuple: 原始图像, 预测掩码
        """
        # 计时开始
        start_time = time.time()

        # 预处理图像
        image, image_tensor = self.preprocess_image(image_path)

        # 分割图像
        mask = self.segment_image(image_tensor, points)

        # 可视化结果
        self.visualize_results(image, mask, output_path, points)

        # 计时结束
        end_time = time.time()
        print(f"分割耗时: {end_time - start_time:.2f} 秒")

        return image, mask


def matplotlib_interactive_point_selection(image_path):
    """
    使用matplotlib进行交互式点选择

    参数:
        image_path (str): 图像路径

    返回:
        list: 点列表，每个点是一个形如(x,y,label)的元组
    """
    # 读取图像（支持中文路径）
    img_data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 存储点的列表
    points = []

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.set_title("点击添加提示点 - 左键: 前景, 右键: 背景")

    # 创建按钮
    plt.subplots_adjust(bottom=0.2)
    button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
    done_button = Button(button_ax, '完成', color='lightgoldenrodyellow', hovercolor='0.975')

    # 按钮回调函数
    def on_button_click(event):
        plt.close(fig)

    done_button.on_clicked(on_button_click)

    # 鼠标点击回调函数
    def on_click(event):
        if event.inaxes != ax:
            return

        if event.button == 1:  # 左键点击
            points.append((int(event.xdata), int(event.ydata), 1))  # 前景点
            ax.plot(event.xdata, event.ydata, 'go', markersize=10)
            print(f"添加前景点: ({int(event.xdata)}, {int(event.ydata)})")
        elif event.button == 3:  # 右键点击
            points.append((int(event.xdata), int(event.ydata), 0))  # 背景点
            ax.plot(event.xdata, event.ydata, 'ro', markersize=10)
            print(f"添加背景点: ({int(event.xdata)}, {int(event.ydata)})")

        fig.canvas.draw()

    # 连接事件
    fig.canvas.mpl_connect('button_press_event', on_click)

    print("点击图像添加提示点:")
    print("  左键点击: 添加前景点 (绿色)")
    print("  右键点击: 添加背景点 (红色)")
    print("  点击'完成'按钮或关闭窗口: 结束添加")

    plt.show()

    print(f"添加了 {len(points)} 个点")
    return points


# def main(): #对单张图片的处理
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

#     # 创建Tkinter根窗口但不显示
#     root = tk.Tk()
#     root.withdraw()

#     # 选择图像文件
#     print("请选择要分割的医学图像...")
#     image_path = filedialog.askopenfilename(
#         title="选择医学图像",
#         filetypes=[("图像文件", "*.jpg *.jpeg *.png")]
#     )
#     plt.rcParams['axes.unicode_minus'] = False  # 显示负号
#     """主函数"""
#     if not image_path:
#         print("未选择图像，程序退出")
#         return

#     print(f"已选择图像: {image_path}")

#     # 检查模型是否存在
#     if not os.path.exists(MODEL_PATH):
#         print(f"错误: 模型不存在: {MODEL_PATH}")
#         print("请在脚本开头的配置参数中设置正确的模型路径")
#         input("按Enter键退出...")
#         return

#     # 设置输出路径
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
#     output_path = os.path.join(OUTPUT_DIR, f"{base_name}_segmentation.png")

#     # 确保输出目录存在
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     # 创建分割器
#     segmenter = MedicalImageSegmenter(MODEL_PATH, MODEL_TYPE)

#     # 获取点
#     points = None
#     if INTERACTIVE_MODE:
#         points = matplotlib_interactive_point_selection(image_path)
#     else:
#         # 默认使用中心点
#         img_data = np.fromfile(image_path, dtype=np.uint8)
#         image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
#         h, w = image.shape[:2]
#         points = [(w // 2, h // 2, 1)]

#     # 处理图像
#     segmenter.process_image(image_path, output_path, points)

#     print("分割完成!")
#     print(f"结果已保存到: {output_path}")
#     print("请关闭所有图像窗口继续...")



def get_all_image_paths(input_dir): 
    """递归遍历文件夹，找到所有图片文件的路径"""
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def main(): #对一整个文件夹的处理
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型不存在: {MODEL_PATH}")
        print("请在脚本开头的配置参数中设置正确的模型路径")
        return

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 创建分割器
    segmenter = MedicalImageSegmenter(MODEL_PATH, MODEL_TYPE)

    # 获取所有图片路径
    image_paths = get_all_image_paths(INPUT_DIR)

    # 分批处理图像，每批处理 100 张
    batch_size = 100
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        for image_path in tqdm(batch_paths, desc=f"处理图像 (第{i // batch_size + 1}批)", unit="图像"):
            print(f"正在处理图像: {image_path}")

            # 使用默认的中心点
            img_data = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            points = [(w // 2, h // 2, 1)]  # 默认使用中心点

            # 处理图像
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            relative_path = os.path.relpath(image_path, INPUT_DIR)
            output_path = os.path.join(OUTPUT_DIR, os.path.splitext(relative_path)[0] + "_segmentation.png")

            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            segmenter.process_image(image_path, output_path, points)

        # 每处理完一批，释放内存
        del segmenter
        torch.cuda.empty_cache()  # 如果使用 GPU，释放 GPU 内存
        segmenter = MedicalImageSegmenter(MODEL_PATH, MODEL_TYPE)

    print("所有图像分割完成！")


if __name__ == "__main__":
    main()