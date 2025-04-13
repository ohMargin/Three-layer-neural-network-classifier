import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import tarfile
import shutil

def download_cifar10(data_dir='./data'):
    """
    下载CIFAR-10数据集
    
    参数:
        data_dir: 数据存储目录
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # CIFAR-10数据集URL
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    
    # 如果文件不存在，下载数据
    if not os.path.exists(filename):
        print("正在下载CIFAR-10数据集...")
        urlretrieve(url, filename)
        print("下载完成!")
    
    # 如果解压后的文件夹不存在，解压数据
    extracted_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir):
        print("正在解压数据...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("解压完成!")
    
    return extracted_dir

def load_cifar10_batch(file_path):
    """
    加载CIFAR-10批次数据
    
    参数:
        file_path: 批次文件路径
        
    返回:
        图像数据和标签
    """
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    
    # 获取图像数据和标签
    images = data_dict[b'data']
    labels = data_dict[b'labels']
    
    # 将图像数据转换为(N, H, W, C)格式
    images = images.reshape(images.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    
    return images, np.array(labels)

def load_cifar10_data(data_dir='./data', normalize=True, flatten=True, validation_size=5000):
    """
    加载CIFAR-10全部数据
    
    参数:
        data_dir: 数据存储目录
        normalize: 是否将像素值归一化到[0,1]
        flatten: 是否将图像展平为一维向量
        validation_size: 验证集大小
        
    返回:
        训练数据、训练标签、验证数据、验证标签、测试数据、测试标签
    """
    # 下载并解压数据（如果需要）
    data_path = download_cifar10(data_dir)
    
    # 加载训练批次
    train_images = []
    train_labels = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_path, f'data_batch_{i}')
        images, labels = load_cifar10_batch(batch_file)
        train_images.append(images)
        train_labels.append(labels)
    
    # 合并训练批次
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    
    # 从训练集中分离验证集
    val_images = train_images[-validation_size:]
    val_labels = train_labels[-validation_size:]
    train_images = train_images[:-validation_size]
    train_labels = train_labels[:-validation_size]
    
    # 加载测试集
    test_file = os.path.join(data_path, 'test_batch')
    test_images, test_labels = load_cifar10_batch(test_file)
    
    # 归一化
    if normalize:
        train_images = train_images.astype('float32') / 255.0
        val_images = val_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
    
    # 展平
    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        val_images = val_images.reshape(val_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    
    print("数据集大小:")
    print(f"训练集: {train_images.shape[0]} 样本")
    print(f"验证集: {val_images.shape[0]} 样本")
    print(f"测试集: {test_images.shape[0]} 样本")
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def get_cifar10_class_names():
    """返回CIFAR-10类别名称"""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

def get_cifar10_class_names_zh():
    """返回CIFAR-10中文类别名称"""
    return [
        '飞机', '汽车', '鸟', '猫', '鹿',
        '狗', '青蛙', '马', '船', '卡车'
    ]

def plot_loss_and_accuracy(train_loss_history, val_loss_history, val_accuracy_history, save_path=None):
    """
    绘制训练过程中的损失和准确率曲线
    
    参数:
        train_loss_history: 训练损失历史
        val_loss_history: 验证损失历史
        val_accuracy_history: 验证准确率历史
        save_path: 保存图表的路径
    """
    epochs = range(1, len(train_loss_history) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss_history, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy_history, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def visualize_weights(model, save_path=None):
    """
    可视化模型权重
    
    参数:
        model: 训练好的模型
        save_path: 保存图表的路径
    """
    # 可视化第一层权重（从输入到隐藏层）
    W1 = model.W1
    
    # 假设输入是CIFAR-10的展平图像（3072 = 32x32x3）
    # 重塑每个神经元的权重为图像形状
    n_neurons = min(64, model.hidden_size)  # 最多显示64个神经元
    rows = int(np.ceil(np.sqrt(n_neurons)))
    cols = int(np.ceil(n_neurons / rows))
    
    plt.figure(figsize=(15, 15))
    
    for i in range(n_neurons):
        if i < model.hidden_size:
            # 重塑权重为3x32x32（彩色图像）
            weight = W1[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
            
            # 归一化到[0,1]以便于可视化
            weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-10)
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(weight)
            plt.axis('off')
            plt.title(f'Neuron {i+1}')
    
    plt.suptitle('First Layer Weights Visualization', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def visualize_model_parameters(model, save_dir='parameter_visualizations', use_chinese=True):
    """
    全面可视化神经网络模型参数，包括权重分布、特征检测器、类别激活模式等
    
    参数:
        model: 训练好的三层神经网络模型
        save_dir: 保存可视化结果的目录
        use_chinese: 是否使用中文类别名称
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置类别名称
    if use_chinese:
        cifar10_classes = get_cifar10_class_names_zh()
    else:
        cifar10_classes = get_cifar10_class_names()
    
    # 设置matplotlib使用支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    
    # 设置可视化样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 提取模型参数
    W1 = model.W1  # 第一层权重 (3072, hidden_size)
    b1 = model.b1  # 第一层偏置 (hidden_size,)
    W2 = model.W2  # 第二层权重 (hidden_size, 10)
    b2 = model.b2  # 第二层偏置 (10,)
    
    # 1. 可视化第一层权重矩阵中的神经元特征检测器
    n_neurons = min(64, model.hidden_size)  # 最多显示64个神经元
    rows = int(np.ceil(np.sqrt(n_neurons)))
    cols = int(np.ceil(np.sqrt(n_neurons)))
    
    plt.figure(figsize=(20, 20))
    
    # 找出权重范数最大的神经元
    neuron_norms = np.linalg.norm(W1, axis=0)
    top_neurons_idx = np.argsort(-neuron_norms)[:n_neurons]
    
    for i, neuron_idx in enumerate(top_neurons_idx):
        if i < n_neurons:
            # 重塑权重为3x32x32（彩色图像）
            weight = W1[:, neuron_idx].reshape(3, 32, 32).transpose(1, 2, 0)
            
            # 归一化到[0,1]以便于可视化
            weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-10)
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(weight)
            plt.axis('off')
            plt.title(f'神经元 #{neuron_idx}\n范数: {neuron_norms[neuron_idx]:.2f}')
    
    plt.suptitle('重要神经元的特征检测器可视化', fontsize=24)
    plt.tight_layout()
    first_layer_vis_path = os.path.join(save_dir, '1_feature_detectors.png')
    plt.savefig(first_layer_vis_path)
    plt.close()
    
    # 2. 可视化第一层权重的统计分布
    plt.figure(figsize=(15, 10))
    
    # 计算每个输入像素位置对所有神经元的重要性
    pixel_importance = np.linalg.norm(W1, axis=1).reshape(3, 32, 32).transpose(1, 2, 0)
    pixel_importance = (pixel_importance - pixel_importance.min()) / (pixel_importance.max() - pixel_importance.min())
    
    # 按通道划分的像素重要性
    # 修复: 正确计算RGB通道的重要性
    input_size = W1.shape[0]
    channel_size = input_size // 3
    r_importance = np.linalg.norm(W1[:channel_size].reshape(32, 32, -1), axis=2)
    g_importance = np.linalg.norm(W1[channel_size:2*channel_size].reshape(32, 32, -1), axis=2)
    b_importance = np.linalg.norm(W1[2*channel_size:].reshape(32, 32, -1), axis=2)
    
    # 可视化权重分布
    plt.subplot(2, 3, 1)
    plt.hist(W1.flatten(), bins=100, alpha=0.7, color='blue')
    plt.title('W1权重分布')
    plt.xlabel('权重值')
    plt.ylabel('频率')
    plt.axvline(x=0, color='red', linestyle='--')
    
    plt.subplot(2, 3, 2)
    plt.hist(W2.flatten(), bins=100, alpha=0.7, color='green')
    plt.title('W2权重分布')
    plt.xlabel('权重值')
    plt.ylabel('频率')
    plt.axvline(x=0, color='red', linestyle='--')
    
    # 可视化偏置分布
    plt.subplot(2, 3, 3)
    plt.hist(b1, bins=50, alpha=0.7, color='purple')
    plt.title('b1偏置分布')
    plt.xlabel('偏置值')
    plt.ylabel('频率')
    plt.axvline(x=0, color='red', linestyle='--')
    
    # 可视化像素重要性
    plt.subplot(2, 3, 4)
    plt.imshow(pixel_importance)
    plt.title('像素重要性热力图')
    plt.colorbar(label='重要性')
    plt.axis('off')
    
    # 可视化通道重要性
    plt.subplot(2, 3, 5)
    width = 0.3
    x = np.arange(32)
    plt.bar(x, r_importance, width, color='red', alpha=0.7, label='R通道')
    plt.bar(x + width, g_importance, width, color='green', alpha=0.7, label='G通道')
    plt.bar(x + 2*width, b_importance, width, color='blue', alpha=0.7, label='B通道')
    plt.title('通道重要性')
    plt.xlabel('像素位置(X轴)')
    plt.ylabel('重要性')
    plt.legend()
    
    # 权重和偏置的统计信息
    plt.subplot(2, 3, 6)
    stats = {
        'W1平均值': np.mean(W1),
        'W1标准差': np.std(W1),
        'W1范数': np.linalg.norm(W1),
        'W2平均值': np.mean(W2),
        'W2标准差': np.std(W2),
        'W2范数': np.linalg.norm(W2),
        'b1平均值': np.mean(b1),
        'b2平均值': np.mean(b2)
    }
    plt.axis('off')
    plt.text(0.1, 0.9, '\n'.join([f"{k}: {v:.4f}" for k, v in stats.items()]), 
             fontsize=12, va='top')
    plt.title('参数统计信息')
    
    plt.suptitle('神经网络参数统计分析', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    stats_vis_path = os.path.join(save_dir, '2_parameter_statistics.png')
    plt.savefig(stats_vis_path)
    plt.close()
    
    # 3. 可视化第二层权重（类别权重）
    plt.figure(figsize=(15, 12))
    
    # 3.1 类别权重热力图
    plt.subplot(2, 2, 1)
    plt.imshow(W2, cmap='viridis', aspect='auto')
    plt.colorbar(label='权重')
    plt.title('输出层权重矩阵')
    plt.xlabel('类别')
    plt.ylabel('隐藏层神经元')
    plt.xticks(range(10), cifar10_classes, rotation=45)
    
    # 3.2 类别偏置条形图
    plt.subplot(2, 2, 2)
    plt.bar(range(10), b2, color='teal')
    plt.title('输出层偏置值')
    plt.xlabel('类别')
    plt.ylabel('偏置值')
    plt.xticks(range(10), cifar10_classes, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 3.3 类别间的相似度矩阵（基于权重）
    similarity_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            similarity_matrix[i, j] = np.dot(W2[:, i], W2[:, j]) / (np.linalg.norm(W2[:, i]) * np.linalg.norm(W2[:, j]))
    
    plt.subplot(2, 2, 3)
    im = plt.imshow(similarity_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar(im, label='余弦相似度')
    plt.title('类别间余弦相似度')
    plt.xticks(range(10), cifar10_classes, rotation=45)
    plt.yticks(range(10), cifar10_classes)
    
    # 3.4 每个类别最重要的特征检测器
    plt.subplot(2, 2, 4)
    class_max_features = np.zeros((10, 5))
    for i in range(10):
        # 找出对该类别贡献最大的5个隐藏层神经元
        top_neurons = np.argsort(-W2[:, i])[:5]
        class_max_features[i] = top_neurons
    
    plt.imshow(class_max_features, cmap='tab20')
    plt.colorbar(label='神经元索引')
    plt.title('每个类别最重要的5个神经元')
    plt.xlabel('神经元重要性排名')
    plt.ylabel('类别')
    plt.yticks(range(10), cifar10_classes)
    
    plt.suptitle('输出层权重分析（类别特征分析）', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    class_vis_path = os.path.join(save_dir, '3_class_weight_analysis.png')
    plt.savefig(class_vis_path)
    plt.close()
    
    # 4. 可视化每个类别的特征模式
    plt.figure(figsize=(20, 15))
    
    for i in range(10):
        plt.subplot(3, 4, i+1)
        
        # 获取该类别的权重
        class_weights = W2[:, i]
        
        # 选取对该类别贡献最大的神经元
        top_neuron_idx = np.argmax(class_weights)
        
        # 重建这个神经元在输入空间的视觉模式
        input_pattern = W1[:, top_neuron_idx].reshape(3, 32, 32).transpose(1, 2, 0)
        # 归一化到[0,1]以便于可视化
        input_pattern = (input_pattern - input_pattern.min()) / (input_pattern.max() - input_pattern.min() + 1e-10)
        
        plt.imshow(input_pattern)
        plt.title(f'{cifar10_classes[i]}\n神经元 #{top_neuron_idx}')
        plt.axis('off')
    
    plt.suptitle('每个类别最活跃的视觉模式', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    class_patterns_path = os.path.join(save_dir, '4_class_visual_patterns.png')
    plt.savefig(class_patterns_path)
    plt.close()
    
    # 5. 保存一个总结文本文件
    summary_path = os.path.join(save_dir, 'parameter_analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("==== 神经网络参数分析摘要 ====\n\n")
        f.write(f"模型结构:\n")
        f.write(f"- 输入大小: {model.input_size}\n")
        f.write(f"- 隐藏层大小: {model.hidden_size}\n")
        f.write(f"- 输出大小: {model.output_size}\n\n")
        
        f.write(f"参数统计:\n")
        f.write(f"- W1形状: {W1.shape}, 参数数量: {W1.size}\n")
        f.write(f"- b1形状: {b1.shape}, 参数数量: {b1.size}\n")
        f.write(f"- W2形状: {W2.shape}, 参数数量: {W2.size}\n")
        f.write(f"- b2形状: {b2.shape}, 参数数量: {b2.size}\n")
        f.write(f"- 总参数数量: {W1.size + b1.size + W2.size + b2.size}\n\n")
        
        f.write(f"权重统计信息:\n")
        f.write(f"- W1平均值: {np.mean(W1):.6f}, 标准差: {np.std(W1):.6f}\n")
        f.write(f"- W1最小值: {np.min(W1):.6f}, 最大值: {np.max(W1):.6f}\n")
        f.write(f"- W2平均值: {np.mean(W2):.6f}, 标准差: {np.std(W2):.6f}\n")
        f.write(f"- W2最小值: {np.min(W2):.6f}, 最大值: {np.max(W2):.6f}\n\n")
        
        f.write(f"特征分析:\n")
        
        # 计算每个类别最重要的神经元
        for i in range(10):
            top_neurons = np.argsort(-W2[:, i])[:3]
            f.write(f"- 类别 '{cifar10_classes[i]}' 最重要的神经元: {top_neurons}\n")
        
        f.write("\n类别相似度分析:\n")
        # 寻找最相似的类别对
        max_sim = -1
        max_pair = (0, 0)
        for i in range(10):
            for j in range(i+1, 10):
                if similarity_matrix[i, j] > max_sim:
                    max_sim = similarity_matrix[i, j]
                    max_pair = (i, j)
        
        f.write(f"- 最相似的类别对: '{cifar10_classes[max_pair[0]]}' 和 '{cifar10_classes[max_pair[1]]}'\n")
        f.write(f"- 相似度: {max_sim:.4f}\n\n")
        
        f.write("参数可视化文件:\n")
        f.write(f"1. 特征检测器可视化: {first_layer_vis_path}\n")
        f.write(f"2. 参数统计分析: {stats_vis_path}\n")
        f.write(f"3. 类别权重分析: {class_vis_path}\n")
        f.write(f"4. 类别视觉模式: {class_patterns_path}\n")
    
    print(f"模型参数可视化完成，所有结果已保存到目录: {save_dir}")
    return save_dir

def visualize_weights_enhanced(model, save_path=None):
    """
    增强版的模型权重可视化函数，整合了简单版和高级版
    
    参数:
        model: 训练好的模型
        save_path: 保存路径（如果是目录，则启用全面可视化）
    """
    if save_path and os.path.isdir(save_path) or (save_path and '.' not in os.path.basename(save_path)):
        # 路径是目录，或者路径不包含扩展名，认为是目录 - 使用全面可视化
        save_dir = save_path
        visualize_model_parameters(model, save_dir)
        return save_dir
    else:
        # 使用传统简单可视化
        return visualize_weights(model, save_path)

def get_mini_batches(X, y, batch_size):
    """
    将数据集划分为小批次
    
    参数:
        X: 输入特征
        y: 标签
        batch_size: 批次大小
        
    返回:
        批次列表
    """
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # 计算完整批次数
    n_batches = n_samples // batch_size
    
    # 创建批次
    mini_batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        mini_batches.append((X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]))
    
    # 如果有剩余样本，创建最后一个小批次
    if n_samples % batch_size != 0:
        start_idx = n_batches * batch_size
        mini_batches.append((X_shuffled[start_idx:], y_shuffled[start_idx:]))
    
    return mini_batches

def cross_entropy_loss(y_pred, y_true):
    """
    计算交叉熵损失
    
    参数:
        y_pred: 预测概率，形状为(batch_size, num_classes)
        y_true: 真实标签，形状为(batch_size, num_classes)或(batch_size,)
        
    返回:
        交叉熵损失值
    """
    batch_size = y_pred.shape[0]
    
    # 如果y_true不是one-hot编码，转换为one-hot编码
    if len(y_true.shape) == 1:
        num_classes = y_pred.shape[1]
        y_one_hot = np.zeros((batch_size, num_classes))
        y_one_hot[np.arange(batch_size), y_true] = 1
        y_true = y_one_hot
    
    # 防止数值不稳定性，避免log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # 计算交叉熵损失
    loss = -np.sum(y_true * np.log(y_pred)) / batch_size
    
    return loss

def apply_data_augmentation(images, augmentation_params=None):
    """
    应用数据增强到图像数据
    
    参数:
        images: 图像数据，形状为(N, H, W, C)
        augmentation_params: 数据增强参数字典，可包含以下选项：
            - horizontal_flip: 是否应用水平翻转 (布尔值)
            - vertical_flip: 是否应用垂直翻转 (布尔值)
            - rotation_range: 随机旋转的角度范围，例如10表示[-10, 10]度
            - width_shift_range: 宽度平移范围，相对于总宽度的比例
            - height_shift_range: 高度平移范围，相对于总高度的比例
            - zoom_range: 随机缩放范围，例如0.1表示[0.9, 1.1]的缩放比例
            - brightness_range: 亮度调整范围，例如0.1表示[0.9, 1.1]的亮度比例
    
    返回:
        增强后的图像数据
    """
    if augmentation_params is None:
        # 默认不进行任何增强
        return images
    
    # 获取图像数量
    n_images = images.shape[0]
    
    # 创建增强后的图像数组
    augmented_images = np.copy(images)
    
    # 应用水平翻转
    if augmentation_params.get('horizontal_flip', False):
        # 随机为50%的图像应用水平翻转
        flip_indices = np.random.choice(n_images, n_images // 2, replace=False)
        augmented_images[flip_indices] = augmented_images[flip_indices, :, ::-1, :]
    
    # 应用垂直翻转
    if augmentation_params.get('vertical_flip', False):
        # 随机为50%的图像应用垂直翻转
        flip_indices = np.random.choice(n_images, n_images // 2, replace=False)
        augmented_images[flip_indices] = augmented_images[flip_indices, ::-1, :, :]
    
    # 应用随机旋转
    rotation_range = augmentation_params.get('rotation_range', 0)
    if rotation_range > 0:
        for i in range(n_images):
            # 随机生成旋转角度
            angle = np.random.uniform(-rotation_range, rotation_range)
            if angle != 0:
                # 应用旋转
                # 这里我们使用scipy的旋转函数
                from scipy.ndimage import rotate
                augmented_images[i] = rotate(augmented_images[i], angle, axes=(0, 1), reshape=False)
    
    # 应用随机宽度平移
    width_shift_range = augmentation_params.get('width_shift_range', 0)
    if width_shift_range > 0:
        for i in range(n_images):
            # 随机生成平移量
            shift = np.random.uniform(-width_shift_range, width_shift_range) * images.shape[2]
            if shift != 0:
                # 应用平移
                from scipy.ndimage import shift as scipy_shift
                augmented_images[i] = scipy_shift(augmented_images[i], (0, shift, 0), mode='nearest')
    
    # 应用随机高度平移
    height_shift_range = augmentation_params.get('height_shift_range', 0)
    if height_shift_range > 0:
        for i in range(n_images):
            # 随机生成平移量
            shift = np.random.uniform(-height_shift_range, height_shift_range) * images.shape[1]
            if shift != 0:
                # 应用平移
                from scipy.ndimage import shift as scipy_shift
                augmented_images[i] = scipy_shift(augmented_images[i], (shift, 0, 0), mode='nearest')
    
    # 应用随机缩放
    zoom_range = augmentation_params.get('zoom_range', 0)
    if zoom_range > 0:
        from scipy.ndimage import zoom as scipy_zoom
        for i in range(n_images):
            # 随机生成缩放因子
            zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
            if zoom_factor != 1:
                # 应用缩放
                # 为了保持图像大小不变，我们需要计算缩放后图像的中心部分
                h, w, c = augmented_images[i].shape
                zoomed_image = scipy_zoom(augmented_images[i], (zoom_factor, zoom_factor, 1), mode='nearest')
                
                # 如果缩放后图像变大，裁剪中心部分
                if zoom_factor > 1:
                    zh, zw, _ = zoomed_image.shape
                    start_h = (zh - h) // 2
                    start_w = (zw - w) // 2
                    zoomed_image = zoomed_image[start_h:start_h+h, start_w:start_w+w, :]
                # 如果缩放后图像变小，填充到原始大小
                else:
                    zh, zw, _ = zoomed_image.shape
                    pad_h = (h - zh) // 2
                    pad_w = (w - zw) // 2
                    padded_image = np.zeros_like(augmented_images[i])
                    padded_image[pad_h:pad_h+zh, pad_w:pad_w+zw, :] = zoomed_image
                    zoomed_image = padded_image
                
                augmented_images[i] = zoomed_image
    
    # 应用亮度调整
    brightness_range = augmentation_params.get('brightness_range', 0)
    if brightness_range > 0:
        for i in range(n_images):
            # 随机生成亮度调整因子
            brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
            if brightness_factor != 1:
                # 应用亮度调整
                augmented_images[i] = augmented_images[i] * brightness_factor
                # 确保像素值在有效范围内
                augmented_images[i] = np.clip(augmented_images[i], 0, 1)
    
    return augmented_images

def load_cifar10_data_with_augmentation(data_dir='./data', normalize=True, flatten=True, validation_size=5000, augmentation_params=None):
    """
    加载CIFAR-10全部数据，并应用数据增强
    
    参数:
        data_dir: 数据存储目录
        normalize: 是否将像素值归一化到[0,1]
        flatten: 是否将图像展平为一维向量
        validation_size: 验证集大小
        augmentation_params: 数据增强参数
        
    返回:
        训练数据、训练标签、验证数据、验证标签、测试数据、测试标签
    """
    # 加载原始数据(不展平，以便进行数据增强)
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_cifar10_data(
        data_dir=data_dir,
        normalize=normalize,
        flatten=False,  # 不展平，以便进行数据增强
        validation_size=validation_size
    )
    
    # 只对训练集应用数据增强
    if augmentation_params is not None:
        print("应用数据增强...")
        print(f"- 水平翻转: {augmentation_params.get('horizontal_flip', False)}")
        print(f"- 垂直翻转: {augmentation_params.get('vertical_flip', False)}")
        print(f"- 旋转范围: ±{augmentation_params.get('rotation_range', 0)}度")
        print(f"- 宽度平移: ±{augmentation_params.get('width_shift_range', 0) * 100}%")
        print(f"- 高度平移: ±{augmentation_params.get('height_shift_range', 0) * 100}%")
        print(f"- 缩放范围: ±{augmentation_params.get('zoom_range', 0) * 100}%")
        print(f"- 亮度调整: ±{augmentation_params.get('brightness_range', 0) * 100}%")
        
        train_images = apply_data_augmentation(train_images, augmentation_params)
    
    # 如果需要展平图像
    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        val_images = val_images.reshape(val_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def apply_advanced_data_augmentation(images, labels, augmentation_params=None):
    """
    应用更先进的数据增强到图像数据
    
    参数:
        images: 图像数据，形状为(N, H, W, C)
        labels: 标签数据，形状为(N,)
        augmentation_params: 数据增强参数字典
    
    返回:
        增强后的图像数据和标签
    """
    if augmentation_params is None:
        return images, labels
    
    # 获取图像数量
    n_images = images.shape[0]
    
    # 创建增强后的图像数组
    augmented_images = np.copy(images)
    augmented_labels = np.copy(labels)
    
    # 决定是否应用混合方法
    use_mixup = augmentation_params.get('mixup', False)
    use_cutmix = augmentation_params.get('cutmix', False)
    
    # 1. 应用基本增强方法(翻转和亮度)
    
    # 应用水平翻转
    if augmentation_params.get('horizontal_flip', False):
        flip_mask = np.random.rand(n_images) < 0.5
        for i in range(n_images):
            if flip_mask[i]:
                augmented_images[i] = augmented_images[i, :, ::-1, :]
    
    # 应用垂直翻转
    if augmentation_params.get('vertical_flip', False):
        flip_mask = np.random.rand(n_images) < 0.5
        for i in range(n_images):
            if flip_mask[i]:
                augmented_images[i] = augmented_images[i, ::-1, :, :]
    
    # 2. 应用随机裁剪和填充
    if augmentation_params.get('random_crop', False):
        crop_size = augmentation_params.get('crop_size', 24)  # 默认裁剪到24x24
        h, w = images.shape[1], images.shape[2]
        pad_h = max(0, h - crop_size)
        pad_w = max(0, w - crop_size)
        
        for i in range(n_images):
            if np.random.rand() < 0.5:  # 50%的概率应用裁剪
                # 随机选择裁剪起始点
                top = np.random.randint(0, pad_h + 1)
                left = np.random.randint(0, pad_w + 1)
                
                # 裁剪图像
                cropped = augmented_images[i, top:top+crop_size, left:left+crop_size, :]
                
                # 填充回原来的大小
                resized = np.zeros_like(augmented_images[i])
                # 使用最近邻插值进行简单的大小调整
                for c in range(3):  # 对每个颜色通道
                    for y in range(h):
                        for x in range(w):
                            y_source = int(y * crop_size / h)
                            x_source = int(x * crop_size / w)
                            resized[y, x, c] = cropped[y_source, x_source, c]
                
                augmented_images[i] = resized
    
    # 3. 应用颜色抖动(HSV空间)
    if augmentation_params.get('color_jitter', False):
        hue_shift = augmentation_params.get('hue_shift', 0.1)
        saturation_scale = augmentation_params.get('saturation_scale', 0.2)
        value_scale = augmentation_params.get('value_scale', 0.2)
        
        for i in range(n_images):
            if np.random.rand() < 0.5:  # 50%的概率应用颜色抖动
                # RGB到HSV转换
                img = augmented_images[i].copy()
                img = np.clip(img, 0, 1)
                
                # 简易RGB到HSV转换
                max_val = np.max(img, axis=2)
                min_val = np.min(img, axis=2)
                delta = max_val - min_val
                
                # 初始化HSV数组
                hsv = np.zeros_like(img)
                
                # Value (V) 通道
                hsv[:, :, 2] = max_val
                
                # Saturation (S) 通道
                hsv[:, :, 1] = np.where(max_val != 0, delta / max_val, 0)
                
                # Hue (H) 通道
                mask_r = (max_val == img[:, :, 0]) & (delta != 0)
                mask_g = (max_val == img[:, :, 1]) & (delta != 0)
                mask_b = (max_val == img[:, :, 2]) & (delta != 0)
                
                hsv[:, :, 0] = np.zeros_like(max_val)
                hsv[mask_r, 0] = ((img[mask_r, 1] - img[mask_r, 2]) / delta[mask_r]) % 6
                hsv[mask_g, 0] = ((img[mask_g, 2] - img[mask_g, 0]) / delta[mask_g]) + 2
                hsv[mask_b, 0] = ((img[mask_b, 0] - img[mask_b, 1]) / delta[mask_b]) + 4
                hsv[:, :, 0] = hsv[:, :, 0] / 6
                
                # 应用颜色抖动
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-hue_shift, hue_shift)) % 1.0
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(1-saturation_scale, 1+saturation_scale), 0, 1)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(1-value_scale, 1+value_scale), 0, 1)
                
                # HSV到RGB转换
                h = hsv[:, :, 0] * 6.0
                s = hsv[:, :, 1]
                v = hsv[:, :, 2]
                
                h_i = np.floor(h).astype(np.int32)
                f = h - h_i
                p = v * (1 - s)
                q = v * (1 - s * f)
                t = v * (1 - s * (1 - f))
                
                rgb = np.zeros_like(hsv)
                
                for j in range(6):
                    mask = (h_i == j)
                    if j == 0:
                        rgb[mask, 0] = v[mask]
                        rgb[mask, 1] = t[mask]
                        rgb[mask, 2] = p[mask]
                    elif j == 1:
                        rgb[mask, 0] = q[mask]
                        rgb[mask, 1] = v[mask]
                        rgb[mask, 2] = p[mask]
                    elif j == 2:
                        rgb[mask, 0] = p[mask]
                        rgb[mask, 1] = v[mask]
                        rgb[mask, 2] = t[mask]
                    elif j == 3:
                        rgb[mask, 0] = p[mask]
                        rgb[mask, 1] = q[mask]
                        rgb[mask, 2] = v[mask]
                    elif j == 4:
                        rgb[mask, 0] = t[mask]
                        rgb[mask, 1] = p[mask]
                        rgb[mask, 2] = v[mask]
                    elif j == 5:
                        rgb[mask, 0] = v[mask]
                        rgb[mask, 1] = p[mask]
                        rgb[mask, 2] = q[mask]
                
                augmented_images[i] = np.clip(rgb, 0, 1)
    
    # 4. 应用随机擦除(Random Erasing)
    if augmentation_params.get('random_erasing', False):
        erase_prob = augmentation_params.get('erase_prob', 0.5)
        erase_area_ratio = augmentation_params.get('erase_area_ratio', (0.02, 0.4))
        erase_aspect_ratio = augmentation_params.get('erase_aspect_ratio', (0.3, 3.0))
        
        h, w = images.shape[1], images.shape[2]
        for i in range(n_images):
            if np.random.rand() < erase_prob:
                # 计算擦除区域的面积
                area_ratio = np.random.uniform(erase_area_ratio[0], erase_area_ratio[1])
                aspect_ratio = np.random.uniform(erase_aspect_ratio[0], erase_aspect_ratio[1])
                
                erase_area = int(h * w * area_ratio)
                erase_h = int(np.sqrt(erase_area / aspect_ratio))
                erase_w = int(np.sqrt(erase_area * aspect_ratio))
                
                # 确保尺寸有效
                erase_h = min(erase_h, h)
                erase_w = min(erase_w, w)
                
                # 随机位置
                top = np.random.randint(0, h - erase_h + 1)
                left = np.random.randint(0, w - erase_w + 1)
                
                # 擦除区域(用随机值填充)
                random_values = np.random.rand(erase_h, erase_w, 3)
                augmented_images[i, top:top+erase_h, left:left+erase_w, :] = random_values
    
    # 5. 应用高斯噪声
    if augmentation_params.get('gaussian_noise', False):
        noise_prob = augmentation_params.get('noise_prob', 0.5)
        noise_scale = augmentation_params.get('noise_scale', 0.05)
        
        for i in range(n_images):
            if np.random.rand() < noise_prob:
                noise = np.random.normal(0, noise_scale, augmented_images[i].shape)
                augmented_images[i] = np.clip(augmented_images[i] + noise, 0, 1)
    
    # 6. 应用Mixup(图像混合)
    if use_mixup:
        mixup_alpha = augmentation_params.get('mixup_alpha', 0.2)
        mixup_prob = augmentation_params.get('mixup_prob', 0.5)
        
        # 混合图像
        for i in range(n_images):
            if np.random.rand() < mixup_prob:
                # 随机选择另一个图像索引
                j = np.random.randint(0, n_images)
                # 生成混合权重
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # 混合图像
                augmented_images[i] = lam * augmented_images[i] + (1 - lam) * augmented_images[j]
                # 混合标签(保留原标签，因为神经网络模型无法处理混合标签)
    
    # 7. 应用CutMix(裁剪混合)
    elif use_cutmix:
        cutmix_alpha = augmentation_params.get('cutmix_alpha', 1.0)
        cutmix_prob = augmentation_params.get('cutmix_prob', 0.5)
        
        h, w = images.shape[1], images.shape[2]
        
        for i in range(n_images):
            if np.random.rand() < cutmix_prob:
                # 随机选择另一个图像索引
                j = np.random.randint(0, n_images)
                
                # 生成混合参数
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                
                # 计算裁剪区域
                cut_ratio = np.sqrt(1. - lam)
                cut_h = int(h * cut_ratio)
                cut_w = int(w * cut_ratio)
                
                # 随机位置
                center_h = np.random.randint(0, h)
                center_w = np.random.randint(0, w)
                
                # 确定裁剪边界
                top = max(0, center_h - cut_h // 2)
                bottom = min(h, center_h + cut_h // 2)
                left = max(0, center_w - cut_w // 2)
                right = min(w, center_w + cut_w // 2)
                
                # 应用CutMix
                augmented_images[i, top:bottom, left:right, :] = augmented_images[j, top:bottom, left:right, :]
                # 保留原标签，神经网络模型无法处理混合标签
    
    # 8. 应用亮度调整
    brightness_range = augmentation_params.get('brightness_range', 0)
    if brightness_range > 0:
        for i in range(n_images):
            if np.random.rand() < 0.5:  # 50%的概率应用亮度调整
                brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
                augmented_images[i] = augmented_images[i] * brightness_factor
                augmented_images[i] = np.clip(augmented_images[i], 0, 1)
    
    return augmented_images, augmented_labels

def load_cifar10_data_with_advanced_augmentation(data_dir='./data', normalize=True, flatten=True, validation_size=5000, augmentation_params=None):
    """
    加载CIFAR-10全部数据，并应用高级数据增强
    
    参数:
        data_dir: 数据存储目录
        normalize: 是否将像素值归一化到[0,1]
        flatten: 是否将图像展平为一维向量
        validation_size: 验证集大小
        augmentation_params: 数据增强参数
        
    返回:
        训练数据、训练标签、验证数据、验证标签、测试数据、测试标签
    """
    # 加载原始数据(不展平，以便进行数据增强)
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_cifar10_data(
        data_dir=data_dir,
        normalize=normalize,
        flatten=False,  # 不展平，以便进行数据增强
        validation_size=validation_size
    )
    
    # 只对训练集应用数据增强
    if augmentation_params is not None:
        print("应用高级数据增强...")
        print(f"- 水平翻转: {augmentation_params.get('horizontal_flip', False)}")
        print(f"- 垂直翻转: {augmentation_params.get('vertical_flip', False)}")
        print(f"- 随机裁剪: {augmentation_params.get('random_crop', False)}")
        print(f"- 颜色抖动: {augmentation_params.get('color_jitter', False)}")
        print(f"- 随机擦除: {augmentation_params.get('random_erasing', False)}")
        print(f"- 高斯噪声: {augmentation_params.get('gaussian_noise', False)}")
        print(f"- Mixup混合: {augmentation_params.get('mixup', False)}")
        print(f"- CutMix裁剪混合: {augmentation_params.get('cutmix', False)}")
        print(f"- 亮度调整: ±{augmentation_params.get('brightness_range', 0) * 100}%")
        
        train_images, train_labels = apply_advanced_data_augmentation(train_images, train_labels, augmentation_params)
    
    # 如果需要展平图像
    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        val_images = val_images.reshape(val_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def apply_batch_augmentation(batch_images, batch_labels, augmentation_params=None):
    """
    对训练批次应用实时数据增强
    
    参数:
        batch_images: 批次图像数据，形状为(batch_size, features)
        batch_labels: 批次标签数据
        augmentation_params: 数据增强参数字典
    
    返回:
        增强后的批次图像和标签
    """
    if augmentation_params is None:
        return batch_images, batch_labels
    
    # 图像需要从展平状态重塑为(H,W,C)格式
    batch_size = batch_images.shape[0]
    images = batch_images.reshape(batch_size, 32, 32, 3)
    
    # 创建增强后的图像数组
    augmented_images = np.copy(images)
    
    # 对批次中的每个图像独立应用数据增强
    for i in range(batch_size):
        # 1. 水平翻转 (概率0.5)
        if augmentation_params.get('horizontal_flip', False) and np.random.rand() < 0.5:
            augmented_images[i] = augmented_images[i, :, ::-1, :]
        
        # 2. 垂直翻转 (概率0.5)
        if augmentation_params.get('vertical_flip', False) and np.random.rand() < 0.5:
            augmented_images[i] = augmented_images[i, ::-1, :, :]
        
        # 3. 随机亮度调整 (概率0.5)
        brightness_range = augmentation_params.get('brightness_range', 0)
        if brightness_range > 0 and np.random.rand() < 0.5:
            brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
            augmented_images[i] = np.clip(augmented_images[i] * brightness_factor, 0, 1)
        
        # 4. 随机擦除 (概率0.5)
        if augmentation_params.get('random_erasing', False) and np.random.rand() < 0.5:
            h, w = 32, 32  # CIFAR-10图像大小
            
            # 随机确定擦除区域大小(擦除面积为原图5%-20%)
            erase_area = np.random.uniform(0.05, 0.2) * h * w
            aspect_ratio = np.random.uniform(0.3, 1.0/0.3)
            
            erase_h = int(np.sqrt(erase_area / aspect_ratio))
            erase_w = int(np.sqrt(erase_area * aspect_ratio))
            
            # 确保尺寸有效
            erase_h = min(erase_h, h-1)
            erase_w = min(erase_w, w-1)
            
            # 随机位置
            top = np.random.randint(0, h - erase_h)
            left = np.random.randint(0, w - erase_w)
            
            # 擦除区域(用随机值或0填充)
            if np.random.rand() < 0.5:
                # 随机值填充
                random_values = np.random.rand(erase_h, erase_w, 3)
                augmented_images[i, top:top+erase_h, left:left+erase_w, :] = random_values
            else:
                # 用0填充(黑色)
                augmented_images[i, top:top+erase_h, left:left+erase_w, :] = 0
        
        # 5. 随机裁剪和调整(概率0.5)
        if augmentation_params.get('random_crop', False) and np.random.rand() < 0.5:
            h, w = 32, 32  # CIFAR-10图像大小
            
            # 随机裁剪比例(裁剪掉原图像的10%-25%)
            crop_ratio = np.random.uniform(0.1, 0.25)
            crop_h = int(h * (1 - crop_ratio))
            crop_w = int(w * (1 - crop_ratio))
            
            # 随机位置
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            
            # 裁剪图像
            cropped = augmented_images[i, top:top+crop_h, left:left+crop_w, :]
            
            # 简单调整回原始大小(上下采样)
            from scipy.ndimage import zoom
            zoom_factor_h = h / crop_h
            zoom_factor_w = w / crop_w
            
            try:
                # 使用scipy的zoom函数调整大小
                zoomed = zoom(cropped, (zoom_factor_h, zoom_factor_w, 1), order=1)
                augmented_images[i] = zoomed
            except:
                # 如果缩放失败，保持原样
                pass
        
        # 6. 高斯噪声 (概率0.5)
        if augmentation_params.get('gaussian_noise', False) and np.random.rand() < 0.5:
            noise_level = np.random.uniform(0.01, 0.05)  # 1%-5%的噪声
            noise = np.random.normal(0, noise_level, augmented_images[i].shape)
            augmented_images[i] = np.clip(augmented_images[i] + noise, 0, 1)
    
    # 将图像展平回原来的形状
    augmented_images = augmented_images.reshape(batch_size, -1)
    
    return augmented_images, batch_labels 