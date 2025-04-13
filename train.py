import numpy as np
import os
import time
from model import ThreeLayerNN
from utils import get_mini_batches, cross_entropy_loss, plot_loss_and_accuracy, apply_batch_augmentation
import matplotlib.pyplot as plt

class Trainer:
    """训练器类，负责模型的训练过程"""
    
    def __init__(self, model, learning_rate=0.01, lr_decay=0.95, lambda_reg=0.001):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型实例
            learning_rate: 初始学习率
            lr_decay: 学习率衰减系数
            lambda_reg: L2正则化系数
        """
        self.model = model
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lambda_reg = lambda_reg
        
        # 记录训练历史
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        
        # 创建保存模型的目录
        os.makedirs('models', exist_ok=True)
        
        # 记录最佳验证准确率
        self.best_val_accuracy = 0
    
    def sgd_update(self, gradients):
        """
        使用SGD更新模型参数
        
        参数:
            gradients: 梯度字典
        """
        self.model.W1 -= self.learning_rate * gradients['W1']
        self.model.b1 -= self.learning_rate * gradients['b1']
        self.model.W2 -= self.learning_rate * gradients['W2']
        self.model.b2 -= self.learning_rate * gradients['b2']
    
    def compute_accuracy(self, X, y):
        """
        计算模型在数据集上的准确率
        
        参数:
            X: 输入特征
            y: 真实标签
            
        返回:
            准确率
        """
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=64, epochs=20, verbose=True, augmentation_params=None):
        """
        训练模型
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            batch_size: 小批量大小
            epochs: 训练轮数
            verbose: 是否打印训练过程
            augmentation_params: 数据增强参数，如果不为None则对每个批次应用实时数据增强
            
        返回:
            训练历史记录
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 学习率衰减
            if epoch > 0:
                self.learning_rate = self.initial_learning_rate * (self.lr_decay ** epoch)
            
            # 获取小批量
            mini_batches = get_mini_batches(X_train, y_train, batch_size)
            
            # 记录每个epoch的总损失
            epoch_loss = 0
            
            # 遍历每个小批量进行训练
            for batch_X, batch_y in mini_batches:
                # 对当前批次应用数据增强（如果启用）
                if augmentation_params is not None:
                    batch_X, batch_y = apply_batch_augmentation(batch_X, batch_y, augmentation_params)
                
                # 前向传播
                y_pred = self.model.forward(batch_X)
                
                # 计算损失
                loss = cross_entropy_loss(y_pred, batch_y)
                
                # 计算L2正则化损失
                if self.lambda_reg > 0:
                    l2_loss = 0.5 * self.lambda_reg * (np.sum(self.model.W1 * self.model.W1) + 
                                                      np.sum(self.model.W2 * self.model.W2))
                    loss += l2_loss
                
                # 反向传播计算梯度
                gradients = self.model.backward(batch_y, lambda_reg=self.lambda_reg)
                
                # 参数更新
                self.sgd_update(gradients)
                
                # 累加损失
                epoch_loss += loss
            
            # 计算平均训练损失
            avg_train_loss = epoch_loss / len(mini_batches)
            self.train_loss_history.append(avg_train_loss)
            
            # 在验证集上评估模型
            val_predictions = self.model.forward(X_val)
            val_loss = cross_entropy_loss(val_predictions, y_val)
            self.val_loss_history.append(val_loss)
            
            # 计算验证集准确率
            val_accuracy = self.compute_accuracy(X_val, y_val)
            self.val_accuracy_history.append(val_accuracy)
            
            # 保存最佳模型
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.model.save_weights(os.path.join('models', 'best_model.npz'))
                if verbose:
                    print(f"保存最佳模型，验证准确率: {val_accuracy:.4f}")
            
            # 打印训练进度
            if verbose:
                end_time = time.time()
                print(f"Epoch {epoch+1}/{epochs} - 时间: {end_time-start_time:.2f}s - "
                      f"训练损失: {avg_train_loss:.4f} - 验证损失: {val_loss:.4f} - "
                      f"验证准确率: {val_accuracy:.4f} - 学习率: {self.learning_rate:.6f}")
        
        # 训练完成，确保加载最佳模型
        self.model.load_weights(os.path.join('models', 'best_model.npz'))
        
        # 返回训练历史
        return {
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'val_accuracy': self.val_accuracy_history
        }
    
    def plot_training_history(self, save_path=None):
        """
        绘制训练历史图表
        
        参数:
            save_path: 保存图表的路径
        """
        plot_loss_and_accuracy(
            self.train_loss_history,
            self.val_loss_history,
            self.val_accuracy_history,
            save_path
        )

def visualize_model_parameters(model, save_dir='parameter_visualizations', cifar10_classes=None):
    """
    全面可视化神经网络模型参数，包括权重分布、特征检测器、类别激活模式等
    
    参数:
        model: 训练好的三层神经网络模型
        save_dir: 保存可视化结果的目录
        cifar10_classes: CIFAR-10类别名称，如果为None则使用默认英文名称
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文类别名称
    if cifar10_classes is None:
        cifar10_classes = get_cifar10_class_names()
        
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
    
    # 按通道划分，计算每个通道的像素重要性
    r_importance = np.linalg.norm(W1[:1024].reshape(32, 32), axis=1)
    g_importance = np.linalg.norm(W1[1024:2048].reshape(32, 32), axis=1)
    b_importance = np.linalg.norm(W1[2048:].reshape(32, 32), axis=1)
    
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