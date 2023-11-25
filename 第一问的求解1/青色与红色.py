import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定英文字体为Times New Roman
plt.rcParams['axes.unicode_minus'] = False    # 解决负数坐标轴显示问题
plt.rcParams['font.size'] = 15                # 指定字号

# 解压缩文件
def extract_zip(zip_file_path, extracted_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)

# 计算图像中苹果的数量
def count_apples(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色苹果的HSV范围
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

    # 定义青色苹果的HSV范围
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # 合并红色和青色苹果的掩码
    mask = cv2.bitwise_or(mask_red, mask_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

# 设置文件路径
zip_file_path = 'Attachment 1.zip'
extracted_folder_path = 'extracted_images'

# 解压缩文件
extract_zip(zip_file_path, extracted_folder_path)

# 获取解压后的文件夹和文件
inner_folder_path = os.path.join(extracted_folder_path, os.listdir(extracted_folder_path)[0])
inner_extracted_files = os.listdir(inner_folder_path)

# 计算每张图像中的苹果数量
apple_counts = []
for file in inner_extracted_files:
    file_path = os.path.join(inner_folder_path, file)
    count = count_apples(file_path)
    apple_counts.append(count)

# 绘制直方图
plt.hist(apple_counts, bins=len(set(apple_counts)), color='red', alpha=0.7)
plt.title('苹果数量分布')
plt.xlabel('苹果数量')
plt.ylabel('图像数量')
plt.show()
