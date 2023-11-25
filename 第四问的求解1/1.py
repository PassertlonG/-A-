import zipfile
import os

# Path to the uploaded zip file
zip_file_path = 'Attachment 1.zip'
extracted_folder_path = 'extracted_images'

# Extracting the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

# Listing the extracted files
extracted_files = os.listdir(extracted_folder_path)
extracted_files.sort()  # Sorting the files for better readability
extracted_files

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定英文字体为Times New Roman
plt.rcParams['axes.unicode_minus'] = False    # 解决负数坐标轴显示问题
plt.rcParams['font.size'] = 15                # 指定字号

def improved_apple_area_calculation(image_path):
    """
    改进的方法来计算图片中苹果的面积，考虑到苹果颜色的多样性和背景的影响。

    :param image_path: 图片的路径
    :return: 苹果的二维面积（像素单位）
    """
    # 读取图片
    image = cv2.imread(image_path)
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色和青色苹果的颜色范围
    # 红色苹果
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    # 青色苹果
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # 创建红色和青色的掩码
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask = mask_red1 + mask_red2 + mask_green

    # 应用掩码
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # 转换为灰度图像并应用阈值
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓并计算面积
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    apple_area = max(areas) if areas else 0

    return apple_area

inner_folder_path = os.path.join(extracted_folder_path, os.listdir(extracted_folder_path)[0])
inner_extracted_files = os.listdir(inner_folder_path)

# 计算每张图片中苹果的面积
improved_apple_areas = []
for file in inner_extracted_files:  
    image_path = os.path.join(inner_folder_path, file)
    area = improved_apple_area_calculation(image_path)
    improved_apple_areas.append(area)

# 显示部分计算结果
print(improved_apple_areas)

## 假设的面积到质量的转换比例
area_to_mass_ratio = 0.005  # 示例值，可根据实际情况调整

# 将面积转换为质量
apple_masses = [area * area_to_mass_ratio for area in improved_apple_areas]

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(apple_masses, bins=20, color='green', alpha=0.7)
plt.title('苹果质量分布')
plt.xlabel('质量 (克)')
plt.ylabel('频数')
plt.grid(True)
plt.show()

