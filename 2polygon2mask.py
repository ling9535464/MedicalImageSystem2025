import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob


def read_image_with_chinese_path(file_path):
    """读取中文路径的图像文件"""
    try:
        file_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"图像解码失败: {file_path}")
        return img
    except Exception as e:
        print(f"读取图像时出错: {file_path}, 错误: {e}")
        return None


def save_image_with_chinese_path(file_path, img):
    """保存图像到中文路径"""
    try:
        _, buffer = cv2.imencode(os.path.splitext(file_path)[1], img)
        buffer.tofile(file_path)
        return True
    except Exception as e:
        print(f"保存图像时出错: {file_path}, 错误: {e}")
        return False


def is_text_area_contour(contour, text_area_threshold=150):
    """判断轮廓是否主要位于文字区域

    参数:
    contour - 轮廓
    text_area_threshold - 文字区域的x坐标阈值（小于此值被认为是文字区域）

    返回:
    bool - 如果轮廓中心点在文字区域则返回True
    """
    # 计算轮廓的中心点
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return False

    cX = int(M["m10"] / M["m00"])

    # 如果轮廓中心点x坐标小于阈值，则认为是文字区域的轮廓
    return cX < text_area_threshold


def filter_small_contours(contours, min_area=50):
    """过滤掉面积过小的轮廓"""
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]


def extract_mask_from_annotated_image(input_image_path, output_mask_path, blue_lower=np.array([100, 50, 50]),
                                      blue_upper=np.array([140, 255, 255]), min_contour_area=50,
                                      text_area_threshold=150):
    """从带有蓝色轮廓标注的医疗图像中提取二进制掩码，处理所有蓝色轮廓并过滤文字区域"""
    # 读取带有蓝色轮廓的原始图像
    img = read_image_with_chinese_path(input_image_path)
    if img is None:
        print(f"无法读取图像: {input_image_path}")
        return None

    # 转换为HSV色彩空间，便于提取蓝色区域
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 创建蓝色区域的掩码
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # 对蓝色掩码进行形态学操作，确保轮廓连续
    kernel = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 提取轮廓
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤掉太小的轮廓（可能是噪点）
    contours = filter_small_contours(contours, min_contour_area)

    if not contours:
        print(f"警告: 在图像中未检测到蓝色轮廓: {input_image_path}")
        return np.zeros(img.shape[:2], dtype=np.uint8), [], []

    # 将轮廓分为文字区域轮廓和有效轮廓
    text_area_contours = []
    valid_contours = []

    for contour in contours:
        if is_text_area_contour(contour, text_area_threshold):
            text_area_contours.append(contour)
        else:
            valid_contours.append(contour)

    print(
        f"在图像 {os.path.basename(input_image_path)} 中检测到 {len(contours)} 个蓝色轮廓，其中 {len(valid_contours)} 个有效轮廓，{len(text_area_contours)} 个文字区域轮廓")

    # 创建一个空白图像作为掩码
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 只填充有效轮廓的内部区域
    for contour in valid_contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 应用形态学闭操作以填充掩码中可能的小洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 保存生成的掩码图像
    save_image_with_chinese_path(output_mask_path, mask)

    return mask, valid_contours, text_area_contours


def process_directory(input_dir, output_dir, blue_lower=np.array([100, 50, 50]), blue_upper=np.array([140, 255, 255]), min_contour_area=50, text_area_threshold=150):
    """处理目录中的所有标注图像并生成掩码"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图像文件
    image_files = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.jpeg")) + glob(os.path.join(input_dir, "*.png"))

    if not image_files:
        print(f"警告: 在目录 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件待处理")
    successful_count = 0
    total_valid_contours = 0
    total_text_area_contours = 0

    # 创建一个字典存储每个图像的轮廓信息，用于后续SAM格式转换
    all_contours_dict = {}

    # 处理每个图像
    for input_path in image_files:
        filename = os.path.basename(input_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_mask.png")

        print(f"处理图像: {filename}")

        # 提取掩码
        result = extract_mask_from_annotated_image(
            input_path,
            output_path,
            blue_lower=blue_lower,
            blue_upper=blue_upper,
            min_contour_area=min_contour_area,
            text_area_threshold=text_area_threshold
        )

        if result is None:
            continue

        mask, valid_contours, text_area_contours = result
        all_contours_dict[base_name] = valid_contours  # 只保存有效轮廓
        total_valid_contours += len(valid_contours)
        total_text_area_contours += len(text_area_contours)

        # 可视化对比结果
        img = read_image_with_chinese_path(input_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(18, 6))

        # 绘制原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title('原始标注图像')

        # 绘制掩码
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('生成的掩码（排除文字区域）')

        # 绘制带有彩色轮廓的原始图像，区分显示有效轮廓和被排除的文字区域轮廓
        plt.subplot(1, 3, 3)
        overlay = img_rgb.copy()

        # 绘制文字区域轮廓（红色）
        for i, contour in enumerate(text_area_contours):
            cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 2)  # 红色
            # 添加轮廓编号
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(overlay, f"T{i + 1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 绘制有效轮廓（绿色）
        for i, contour in enumerate(valid_contours):
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)  # 绿色
            # 添加轮廓编号
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(overlay, f"V{i + 1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        plt.imshow(overlay)
        plt.title(f'检测到的轮廓 (绿色: {len(valid_contours)}个有效, 红色: {len(text_area_contours)}个文字区域)')

        # 绘制垂直分隔线表示文字区域阈值
        plt.axvline(x=text_area_threshold, color='cyan', linestyle='--')
        plt.text(text_area_threshold + 5, 50, f'文字区域阈值 x={text_area_threshold}', color='cyan')

        plt.tight_layout()
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.savefig(comparison_path)
        plt.close()

        successful_count += 1

    print(f"成功处理 {successful_count}/{len(image_files)} 个图像")
    print(f"共检测到 {total_valid_contours} 个有效蓝色轮廓区域，{total_text_area_contours} 个文字区域轮廓被排除")

    return all_contours_dict


def convert_masks_to_sam_format(mask_dir, contours_dict=None, original_images_dir=None):
    """将生成的掩码转换为SAM模型所需的格式，支持多区域标注"""
    import json

    sam_data = []
    annotation_id = 1

    if contours_dict:
        # 使用已经提取的轮廓信息
        for base_name, contours in contours_dict.items():
            # 为每个轮廓创建一个标注
            for i, contour in enumerate(contours):
                # 转换轮廓为点列表
                points = contour.reshape(-1, 2).tolist()
                flattened_points = []
                for point in points:
                    flattened_points.extend(point)

                # 创建标注
                x, y, w, h = cv2.boundingRect(contour)
                annotation = {
                    "id": annotation_id,
                    "image_id": base_name,
                    "category_id": 1,  # 类别ID，通常病变为1
                    "segmentation": [flattened_points],
                    "area": cv2.contourArea(contour),
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                    "region_number": i + 1  # 区域编号，从1开始
                }

                # 如果有原始图像目录，添加图像信息
                if original_images_dir:
                    for ext in ['.jpg', '.jpeg', '.png']:
                        potential_path = os.path.join(original_images_dir, base_name + ext)
                        if os.path.exists(potential_path):
                            img = read_image_with_chinese_path(potential_path)
                            if img is not None:
                                annotation["image"] = {
                                    "file_name": os.path.basename(potential_path),
                                    "height": img.shape[0],
                                    "width": img.shape[1]
                                }
                            break

                sam_data.append(annotation)
                annotation_id += 1
    else:
        # 如果没有提供轮廓信息，则从保存的掩码文件中提取
        mask_files = glob(os.path.join(mask_dir, "*_mask.png"))

        for mask_path in mask_files:
            base_name = os.path.basename(mask_path).replace('_mask.png', '')

            # 读取掩码
            mask = read_image_with_chinese_path(mask_path)
            if mask is None:
                continue

            # 如果读取的是彩色图像，转换为灰度
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 为每个轮廓创建一个标注
            for i, contour in enumerate(contours):
                # 转换轮廓为点列表
                points = contour.reshape(-1, 2).tolist()
                flattened_points = []
                for point in points:
                    flattened_points.extend(point)

                # 创建标注
                x, y, w, h = cv2.boundingRect(contour)
                annotation = {
                    "id": annotation_id,
                    "image_id": base_name,
                    "category_id": 1,  # 类别ID，通常病变为1
                    "segmentation": [flattened_points],
                    "area": cv2.contourArea(contour),
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                    "region_number": i + 1  # 区域编号，从1开始
                }

                sam_data.append(annotation)
                annotation_id += 1


def main(original_images_dir=None, input_directory=None):
    # 如果没有指定路径，使用默认路径
    if original_images_dir is None:
        original_images_dir = "./image_dataset/分割/原图"  # 原始图像目录（可选）
    if input_directory is None:
        input_directory = "./image_dataset/分割/标注分割"  # 包含蓝色轮廓标注图像的文件夹

    print(f"原图文件夹: {original_images_dir}")
    print(f"标注分割文件夹: {input_directory}")
    # 配置目录
    output_directory = os.path.join(os.path.dirname(original_images_dir), '掩码分割')  # 存储生成的掩码图像的文件夹
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    # 配置参数
    blue_lower = np.array([100, 50, 50])  # 蓝色HSV下限
    blue_upper = np.array([140, 255, 255])  # 蓝色HSV上限
    min_contour_area = 50  # 最小轮廓面积，用于过滤噪点
    text_area_threshold = 120  # 文字区域的x坐标阈值，小于此值的轮廓被视为文字区域轮廓

    # 处理图像并生成掩码，返回轮廓信息
    contours_dict = process_directory(
        input_directory,
        output_directory,
        blue_lower=blue_lower,
        blue_upper=blue_upper,
        min_contour_area=min_contour_area,
        text_area_threshold=text_area_threshold
    )

    # 转换为SAM格式
    convert_masks_to_sam_format(
        output_directory,
        contours_dict=contours_dict,
        original_images_dir=original_images_dir
    )

    print("处理完成!")


# 主函数
if __name__ == "__main__":
    main()
