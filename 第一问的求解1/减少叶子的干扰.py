import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定英文字体为Times New Roman
plt.rcParams['axes.unicode_minus'] = False    # 解决负数坐标轴显示问题
plt.rcParams['font.size'] = 15                # 指定字号

def extract_zip(zip_file_path, extracted_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)

def count_apples(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色苹果的HSV范围（放宽颜色范围）
    lower_red = np.array([0, 40, 40])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

    # 定义青色苹果的HSV范围（放宽颜色范围）
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    mask = cv2.bitwise_or(mask_red, mask_green)

    # 调整形态学操作参数
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    apple_count = 0
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            eccentricity = np.sqrt(1 - (minoraxis_length/majoraxis_length)**2)

            # 调整面积和偏心率的阈值
            area = cv2.contourArea(contour)
            if area > 50 and eccentricity < 0.85:
                apple_count += 1

    return apple_count

zip_file_path = 'Attachment 1.zip'
extracted_folder_path = 'extracted_images'

extract_zip(zip_file_path, extracted_folder_path)

inner_folder_path = os.path.join(extracted_folder_path, os.listdir(extracted_folder_path)[0])
inner_extracted_files = os.listdir(inner_folder_path)

apple_counts = []
for file in inner_extracted_files:
    file_path = os.path.join(inner_folder_path, file)
    count = count_apples(file_path)
    apple_counts.append(count)

plt.hist(apple_counts, bins=len(set(apple_counts)), color='red', alpha=0.7)
plt.title('苹果数量分布')
plt.xlabel('苹果数量')
plt.ylabel('图像数量')
plt.show()

import random

# 绘制箱型图
plt.figure(figsize=(8, 6))
plt.boxplot(apple_counts, patch_artist=True)
plt.title('苹果数量箱型图')
plt.ylabel('苹果数量')
plt.show()

# 绘制累积分布函数图
sorted_counts = np.sort(apple_counts)
cdf = np.arange(1, len(sorted_counts)+1) / len(sorted_counts)
plt.figure(figsize=(8, 6))
plt.plot(sorted_counts, cdf, marker='.', linestyle='none')
plt.title('苹果数量的累积分布函数')
plt.xlabel('苹果数量')
plt.ylabel('CDF')
plt.show()

# 随机选择几个图像进行展示
sample_indexes = random.sample(range(len(inner_extracted_files)), 5)
sample_files = [inner_extracted_files[i] for i in sample_indexes]
sample_counts = [apple_counts[i] for i in sample_indexes]

plt.figure(figsize=(15, 3))
for i, file in enumerate(sample_files):
    img_path = os.path.join(inner_folder_path, file)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f'Count: {sample_counts[i]}')
    plt.axis('off')
plt.suptitle('Sample Images with Apple Counts')
plt.show()

# 选择一个样本图像进行展示
sample_file = inner_extracted_files[3]  # 选择第一个文件作为样本
sample_img_path = os.path.join(inner_folder_path, sample_file)
sample_image = cv2.imread(sample_img_path)

# 应用颜色范围以识别苹果
hsv_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 40, 40])
upper_red = np.array([10, 255, 255])
mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask = cv2.bitwise_or(mask_red, mask_green)

# 应用形态学操作
kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
mask = cv2.erode(mask, kernel, iterations=1)

# 显示原始图像和处理后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
plt.title('原始图像')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('处理后的图像')
plt.axis('off')
plt.show()


