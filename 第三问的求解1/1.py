import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负数坐标轴显示问题
plt.rcParams['font.size'] = 15                # 设置字体大小

# 解压缩文件
zip_path = 'Attachment 1.zip'
extract_folder = 'unzipped_apple_images'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

    # 获取解压后的文件夹路径
inner_folder_path = os.path.join(extract_folder, os.listdir(extract_folder)[0])
inner_extracted_files = os.listdir(inner_folder_path)


def calculate_apple_ripeness_with_shape_exclusion(image_path):
    """
    计算苹果成熟度的函数，同时考虑形状以排除叶子的干扰。
    参数:
    image_path: 苹果图片的路径。

    返回值:
    ripeness_score: 苹果的成熟度得分，基于红色和青色像素的比例，排除了叶子的影响。
    """
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从BGR转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 定义红色和青色的阈值
    red_lower = np.array([100, 0, 0])
    red_upper = np.array([255, 100, 100])
    green_lower = np.array([0, 100, 0])
    green_upper = np.array([100, 255, 100])

    # 创建红色和青色的掩膜
    red_mask = cv2.inRange(image_rgb, red_lower, red_upper)
    green_mask = cv2.inRange(image_rgb, green_lower, green_upper)

    # 通过形状排除叶子
    # 查找轮廓
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 排除非圆形轮廓
    for contour in contours:
        # 计算轮廓的面积和边界矩形
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # 检查轮廓是否足够圆
        if w > 0 and h > 0:
            aspect_ratio = w / h
            if aspect_ratio > 0.8 and aspect_ratio < 1.2 and area > 100:  # 这里的阈值可以根据实际情况调整
                # 将此轮廓视为苹果，保留其像素
                cv2.drawContours(green_mask, [contour], -1, (255, 255, 255), -1)
            else:
                # 将此轮廓视为叶子，移除其像素
                cv2.drawContours(green_mask, [contour], -1, (0, 0, 0), -1)

    # 计算红色和青色像素的数量（排除了叶子）
    red_count = np.sum(red_mask > 0)
    green_count = np.sum(green_mask > 0)

    # 计算成熟度得分
    ripeness_score = red_count / (red_count + green_count + 1e-6)  # 添加一个小量以防止除以零

    return ripeness_score

# 重新计算每张图片的成熟度得分，考虑形状排除
ripeness_scores_shape_excluded = []
for file in inner_extracted_files:
    image_path = os.path.join(inner_folder_path, file)
    score = calculate_apple_ripeness_with_shape_exclusion(image_path)
    ripeness_scores_shape_excluded.append(score)

# 绘制考虑形状排除后的成熟度分布直方图
plt.hist(ripeness_scores_shape_excluded, bins=20, color='blue')
plt.title('考虑形状排除的苹果成熟度分布直方图')
plt.xlabel('成熟度得分')
plt.ylabel('频数')
plt.show()

# 绘制箱型图
plt.figure(figsize=(8, 6))
plt.boxplot(ripeness_scores_shape_excluded, patch_artist=True)
plt.title('苹果成熟度得分箱型图')
plt.ylabel('成熟度得分')
plt.show()

# 绘制累积分布函数图
sorted_scores = np.sort(ripeness_scores_shape_excluded)
cdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
plt.figure(figsize=(8, 6))
plt.plot(sorted_scores, cdf, marker='.', linestyle='none')
plt.title('苹果成熟度得分的累积分布函数')
plt.xlabel('成熟度得分')
plt.ylabel('CDF')
plt.show()

# 随机选择几个图像进行展示
random_sample_indexes = np.random.choice(len(inner_extracted_files), 3, replace=False)
plt.figure(figsize=(15, 5))

for i, index in enumerate(random_sample_indexes, 1):
    file = inner_extracted_files[index]
    img_path = os.path.join(inner_folder_path, file)
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    score = ripeness_scores_shape_excluded[index]

    plt.subplot(1, 3, i)
    plt.imshow(image)
    plt.title(f'Score: {score:.2f}')
    plt.axis('off')

plt.suptitle('示例图像与其成熟度得分')
plt.show()

