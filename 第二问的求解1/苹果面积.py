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

# 查找成熟苹果的中心点及其大小
def find_mature_apple_centers_and_sizes(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义成熟苹果（红色）的HSV颜色范围
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # 形态学操作来减少噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    sizes = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            sizes.append(cv2.contourArea(contour))

    return centers, sizes

zip_file_path = 'Attachment 1.zip'  # 更改为你的文件路径
extracted_folder_path = 'extracted_images' # 更改为提取路径

# 解压缩文件
extract_zip(zip_file_path, extracted_folder_path)

# 获取解压后的文件夹和文件
inner_folder_path = os.path.join(extracted_folder_path, os.listdir(extracted_folder_path)[0])
inner_extracted_files = os.listdir(inner_folder_path)

# 收集所有图像中成熟苹果的中心点坐标和大小
all_centers = []
all_sizes = []
for file in inner_extracted_files:
    file_path = os.path.join(inner_folder_path, file)
    centers, sizes = find_mature_apple_centers_and_sizes(file_path)
    all_centers.extend(centers)
    all_sizes.extend(sizes)

# 获取图像的高度用于坐标转换
image_height = cv2.imread(os.path.join(inner_folder_path, inner_extracted_files[0])).shape[0]

# 将坐标转换为以左下角为原点的系统
transformed_centers = [(x, image_height - y) for (x, y) in all_centers]

# 调整点的大小以表示苹果的大小
sizes_normalized = [s / max(all_sizes) * 100 for s in all_sizes]  # 归一化大小

# 绘制带有大小表示的二维散点图
x_coords, y_coords = zip(*transformed_centers)
plt.scatter(x_coords, y_coords, s=sizes_normalized, alpha=0.6)
plt.title('成熟苹果的几何坐标及大小散点图')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.gca().invert_yaxis()  # 反转Y轴以符合常规坐标系
plt.show()
