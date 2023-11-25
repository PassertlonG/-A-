import zipfile
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import cv2
import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定英文字体为Times New Roman
plt.rcParams['axes.unicode_minus'] = False    # 解决负数坐标轴显示问题
plt.rcParams['font.size'] = 15                # 指定字号

# File paths for the uploaded zip files
zip_file_path_2 = 'Attachment 2.zip'
zip_file_path_3 = 'Attachment 3.zip'

# Unzipping the files
with zipfile.ZipFile(zip_file_path_2, 'r') as zip_ref:
    zip_ref.extractall("dataset2")

with zipfile.ZipFile(zip_file_path_3, 'r') as zip_ref:
    zip_ref.extractall("dataset3")

# Checking the contents of the unzipped folders
dataset2_contents = os.listdir("dataset2")
dataset3_contents = os.listdir("dataset3")

# Further checking the contents of the nested directories
dataset2_nested_contents = os.listdir("dataset2/Attachment 2")
dataset3_nested_contents = os.listdir("dataset3/Attachment 3")

# 定义一个函数来加载和预处理图像
def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(f'dataset2/Attachment 2/{folder}/*.jpg'):
        img = cv2.imread(filename)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # 调整图像大小
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB
            images.append(img)
    return images

# 加载苹果图像
apple_images = load_images_from_folder('Apple')

# 检查加载的图像数量和一个样本图像
len(apple_images), apple_images[0].shape

# 定义一个函数来计算图像的颜色直方图
def compute_histogram(image, bins=8):
    # 计算每个颜色通道的直方图
    hist = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(3)]
    hist = np.concatenate(hist)
    hist = cv2.normalize(hist, hist).flatten()  # 归一化直方图
    return hist

# 计算苹果图像的特征
apple_features = [compute_histogram(image) for image in apple_images]

# 展示苹果图像的样本
sample_images = apple_images[:5]  # 选择前5个图像作为样本

plt.figure(figsize=(15, 3))
for i, img in enumerate(sample_images, 1):
    plt.subplot(1, 5, i)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle('苹果图像样本')
plt.show()

# 绘制其中一个图像的颜色直方图
sample_image = sample_images[0]  # 选择第一个图像
color = ('r', 'g', 'b')
plt.figure(figsize=(8, 6))
for i, col in enumerate(color):
    hist = cv2.calcHist([sample_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('颜色直方图')
plt.xlabel('像素强度')
plt.ylabel('频数')
plt.show()

# 同样的方法处理其他水果图像
other_fruits = ['Carambola', 'Pear', 'Plum', 'Tomatoes']
other_fruits_features = []
for fruit in other_fruits:
    images = load_images_from_folder(fruit)
    features = [compute_histogram(image) for image in images]
    other_fruits_features.extend(features)

    # 对于每种其他水果，我们将展示它们的图像样本和一个图像的颜色直方图

# 遍历其他水果，并展示每种水果的图像样本和颜色直方图
for fruit in other_fruits:
    # 加载水果图像
    fruit_images = load_images_from_folder(fruit)
    sample_images = fruit_images[:5]  # 选择前5个图像作为样本

    # 展示图像样本
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(sample_images, 1):
        plt.subplot(1, 5, i)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f'{fruit} 图像样本')
    plt.show()

    # 绘制其中一个图像的颜色直方图
    sample_image = sample_images[0]  # 选择第一个图像
    color = ('r', 'g', 'b')
    plt.figure(figsize=(8, 6))
    for i, col in enumerate(color):
        hist = cv2.calcHist([sample_image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title(f'{fruit} 颜色直方图')
    plt.xlabel('像素强度')
    plt.ylabel('频数')
    plt.show()

# 创建标签，苹果为1，其他为0
labels = np.concatenate([np.ones(len(apple_features)), np.zeros(len(other_fruits_features))])

# 合并特征和标签
features = np.array(apple_features + other_fruits_features)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 创建并训练随机森林分类器
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy, report)

# 在测试集上的混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.title('混淆矩阵')
plt.ylabel('真实类别')
plt.xlabel('预测类别')
plt.show()

from sklearn.metrics import roc_curve, auc

# 计算ROC曲线的参数
y_pred_proba = classifier.predict_proba(X_test)[:, 1]  # 获取苹果类别的概率预测
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('接收者操作特征曲线（ROC）')
plt.legend(loc='lower right')
plt.show()

# 加载 "Attachment 3 - 副本.zip" 中的图像
dataset3_path = "dataset3/Attachment 3"
test_images = []
test_image_ids = []
for filename in sorted(glob.glob(f'{dataset3_path}/*.jpg')):
    img = cv2.imread(filename)
    if img is not None:
        img = cv2.resize(img, (64, 64))  # 调整图像大小
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB
        test_images.append(img)
        test_image_ids.append(os.path.basename(filename).split('.')[0])

# 计算测试图像的特征
test_features = [compute_histogram(image) for image in test_images]

# 使用模型进行预测
test_predictions = classifier.predict(test_features)

# 提取识别为苹果的图像ID
apple_ids = [id for id, pred in zip(test_image_ids, test_predictions) if pred == 1]

# 计算步长，以便在直方图上合理地显示标签
step_size = max(len(apple_ids) // 10, 1)  # 每隔step_size个ID显示一个标签，至少显示一个标签

# 为了确保显示的标签是有意义的，我们需要从apple_ids列表中提取相应的标签
displayed_labels = apple_ids[::step_size]

# 绘制调整后的直方图
plt.figure(figsize=(10, 6))
plt.hist(apple_ids, bins=len(displayed_labels), color='green', alpha=0.7)
plt.xlabel('图像ID')
plt.ylabel('频数')
plt.title('识别为苹果的图像ID分布')
plt.xticks(displayed_labels, rotation=45)
plt.show()