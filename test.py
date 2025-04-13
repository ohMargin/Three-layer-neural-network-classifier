import numpy as np
import time
import os
from model import ThreeLayerNN
from utils import get_cifar10_class_names, visualize_weights

def test_model(model, X_test, y_test, batch_size=128):
    """
    在测试集上评估模型性能
    
    参数:
        model: 训练好的模型
        X_test: 测试数据特征
        y_test: 测试数据标签
        batch_size: 批次大小
        
    返回:
        测试准确率
    """
    # 测试数据样本数
    n_samples = X_test.shape[0]
    
    # 用于存储所有批次的预测结果
    all_predictions = []
    
    # 分批次进行预测以避免内存问题
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X_test[i:end_idx]
        
        # 获取模型预测
        batch_predictions = model.predict(batch_X)
        all_predictions.append(batch_predictions)
    
    # 合并所有批次的预测结果
    predictions = np.concatenate(all_predictions)
    
    # 计算准确率
    accuracy = np.mean(predictions == y_test)
    
    return accuracy

def evaluate_detailed(model, X_test, y_test, batch_size=128):
    """
    详细评估模型性能，包括类别准确率和混淆矩阵
    
    参数:
        model: 训练好的模型
        X_test: 测试数据特征
        y_test: 测试数据标签
        batch_size: 批次大小
        
    返回:
        详细评估结果
    """
    # 获取类别名称
    class_names = get_cifar10_class_names()
    num_classes = len(class_names)
    
    # 测试数据样本数
    n_samples = X_test.shape[0]
    
    # 用于存储所有批次的预测结果
    all_predictions = []
    all_probas = []
    
    # 分批次进行预测以避免内存问题
    start_time = time.time()
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X_test[i:end_idx]
        
        # 获取模型预测概率
        batch_probas = model.forward(batch_X)
        batch_predictions = np.argmax(batch_probas, axis=1)
        
        all_probas.append(batch_probas)
        all_predictions.append(batch_predictions)
    
    # 合并所有批次的预测结果
    predictions = np.concatenate(all_predictions)
    probas = np.concatenate(all_probas)
    
    # 计算总体准确率
    accuracy = np.mean(predictions == y_test)
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for i in range(num_classes):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predictions[class_mask] == i)
            class_accuracies[class_names[i]] = class_acc
    
    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_test)):
        true_label = y_test[i]
        pred_label = predictions[i]
        confusion_matrix[true_label, pred_label] += 1
    
    end_time = time.time()
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': confusion_matrix,
        'inference_time': end_time - start_time
    }

def load_and_test(model_path, X_test, y_test, input_size, hidden_size, output_size, activation='relu'):
    """
    加载保存的模型并在测试集上评估
    
    参数:
        model_path: 模型权重文件路径
        X_test: 测试数据特征
        y_test: 测试数据标签
        input_size: 输入特征维度
        hidden_size: 隐藏层神经元数量
        output_size: 输出类别数量
        activation: 激活函数类型
        
    返回:
        测试结果
    """
    # 创建模型实例
    model = ThreeLayerNN(input_size, hidden_size, output_size, activation)
    
    # 加载保存的权重
    model.load_weights(model_path)
    
    # 测试模型性能
    results = evaluate_detailed(model, X_test, y_test)
    
    # 打印结果
    print(f"测试集准确率: {results['accuracy']:.4f}")
    print("\n每个类别的准确率:")
    for class_name, class_acc in results['class_accuracies'].items():
        print(f"{class_name}: {class_acc:.4f}")
    
    print(f"\n推理时间: {results['inference_time']:.2f}秒")
    
    # 可视化模型权重
    visualize_weights(model, save_path='weight_visualization.png')
    
    return results 