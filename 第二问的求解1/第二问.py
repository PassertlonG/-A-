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

# 查找成熟苹果的中心点
def find_mature_apple_centers(image_path):
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
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

    return centers

zip_file_path = 'Attachment 1.zip'
extracted_folder_path = 'extracted_images'

# 解压缩文件
extract_zip(zip_file_path, extracted_folder_path)

# 获取解压后的文件夹和文件
inner_folder_path = os.path.join(extracted_folder_path, os.listdir(extracted_folder_path)[0])
inner_extracted_files = os.listdir(inner_folder_path)

# 收集所有图像中成熟苹果的中心点坐标
all_centers = []
for file in inner_extracted_files:
    file_path = os.path.join(inner_folder_path, file)
    centers = find_mature_apple_centers(file_path)
    all_centers.extend(centers)

# 获取图像的高度用于坐标转换
image_height = cv2.imread(os.path.join(inner_folder_path, inner_extracted_files[0])).shape[0]

# 将坐标转换为以左下角为原点的系统
transformed_centers = [(x, image_height - y) for (x, y) in all_centers]

# 绘制二维散点图
x_coords, y_coords = zip(*transformed_centers)
plt.scatter(x_coords, y_coords)
plt.title('成熟苹果的几何坐标散点图')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.gca().invert_yaxis()  # 反转Y轴以符合常规坐标系
plt.show()

import seaborn as sns

# 绘制散点图热力图
plt.figure(figsize=(10, 6))
sns.kdeplot(x=x_coords, y=y_coords, cmap="Reds", shade=True, bw_adjust=0.5)
plt.title('成熟苹果的几何坐标热力图')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.gca().invert_yaxis()
plt.show()

# 展示一些示例图像并在图像上标注识别出的成熟苹果的位置
sample_files = inner_extracted_files[:2]  # 选择前3个文件作为示例
plt.figure(figsize=(15, 5))

for i, file in enumerate(sample_files, 1):
    img_path = os.path.join(inner_folder_path, file)
    image = cv2.imread(img_path)
    centers = find_mature_apple_centers(img_path)

    for (x, y) in centers:
        cv2.circle(image, (x, y), 10, (0, 255, 0), 2)

    plt.subplot(1, 3, i)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Image: {file}')
    plt.axis('off')

plt.suptitle('示例图像与识别出的成熟苹果位置')
plt.show()

