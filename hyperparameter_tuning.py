import numpy as np
import time
import os
import json
from model import ThreeLayerNN
from train import Trainer
from utils import get_mini_batches
import itertools
import matplotlib.pyplot as plt

class HyperparameterTuner:
    """超参数调优类"""
    
    def __init__(self, X_train, y_train, X_val, y_val, input_size, output_size):
        """
        初始化超参数调优器
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            input_size: 输入特征维度
            output_size: 输出类别数量
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_size = input_size
        self.output_size = output_size
        
        # 创建结果目录
        os.makedirs('tuning_results', exist_ok=True)
        
        # 存储所有实验结果
        self.results = []
    
    def train_with_hyperparams(self, hidden_size, activation, learning_rate, lr_decay, lambda_reg, batch_size, epochs):
        """
        使用给定超参数训练模型
        
        参数:
            hidden_size: 隐藏层神经元数量
            activation: 激活函数类型
            learning_rate: 学习率
            lr_decay: 学习率衰减系数
            lambda_reg: L2正则化系数
            batch_size: 批次大小
            epochs: 训练轮数
            
        返回:
            训练结果
        """
        # 创建模型
        model = ThreeLayerNN(self.input_size, hidden_size, self.output_size, activation)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            lambda_reg=lambda_reg
        )
        
        # 训练模型
        start_time = time.time()
        history = trainer.train(
            self.X_train, self.y_train, 
            self.X_val, self.y_val,
            batch_size=batch_size, 
            epochs=epochs, 
            verbose=False  # 关闭详细输出，避免在grid search中产生过多输出
        )
        training_time = time.time() - start_time
        
        # 获取最佳验证准确率
        best_val_accuracy = trainer.best_val_accuracy
        
        # 返回结果
        return {
            'hidden_size': hidden_size,
            'activation': activation,
            'learning_rate': learning_rate,
            'lr_decay': lr_decay,
            'lambda_reg': lambda_reg,
            'batch_size': batch_size,
            'epochs': epochs,
            'best_val_accuracy': best_val_accuracy,
            'training_time': training_time,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
    
    def grid_search(self, param_grid, max_evals=None):
        """
        网格搜索超参数
        
        参数:
            param_grid: 超参数网格，字典形式，键为参数名，值为参数候选值列表
            max_evals: 最大评估次数，如果为None则评估所有组合
            
        返回:
            最佳参数和所有结果
        """
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # 如果指定了最大评估次数，随机选择部分组合
        if max_evals is not None and max_evals < len(param_combinations):
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_evals]
        
        # 总组合数
        total_combinations = len(param_combinations)
        print(f"将评估 {total_combinations} 种参数组合")
        
        # 评估每种参数组合
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            
            print(f"\n评估组合 {i+1}/{total_combinations}:")
            for name, value in param_dict.items():
                print(f"  {name}: {value}")
            
            # 训练并评估模型
            result = self.train_with_hyperparams(
                hidden_size=param_dict.get('hidden_size'),
                activation=param_dict.get('activation'),
                learning_rate=param_dict.get('learning_rate'),
                lr_decay=param_dict.get('lr_decay'),
                lambda_reg=param_dict.get('lambda_reg'),
                batch_size=param_dict.get('batch_size'),
                epochs=param_dict.get('epochs')
            )
            
            # 打印结果
            print(f"  验证准确率: {result['best_val_accuracy']:.4f}")
            print(f"  训练时间: {result['training_time']:.2f}秒")
            
            # 保存结果
            self.results.append(result)
        
        # 寻找最佳参数
        best_result = max(self.results, key=lambda x: x['best_val_accuracy'])
        
        # 保存所有结果
        with open(os.path.join('tuning_results', 'hyperparameter_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 打印最佳参数
        print("\n最佳参数:")
        for name in param_names:
            print(f"  {name}: {best_result[name]}")
        print(f"最佳验证准确率: {best_result['best_val_accuracy']:.4f}")
        
        return best_result, self.results
    
    def visualize_results(self, param_of_interest, save_path=None):
        """
        可视化超参数调优结果
        
        参数:
            param_of_interest: 关注的参数名称
            save_path: 保存图表的路径
        """
        if not self.results:
            print("没有可视化的结果，请先运行grid_search")
            return
        
        # 提取参数值和对应的准确率
        param_values = [r[param_of_interest] for r in self.results]
        accuracies = [r['best_val_accuracy'] for r in self.results]
        
        # 按参数值对结果进行分组
        unique_values = set(param_values)
        grouped_accuracies = {val: [] for val in unique_values}
        
        for val, acc in zip(param_values, accuracies):
            grouped_accuracies[val].append(acc)
        
        # 计算每个参数值的平均准确率
        avg_accuracies = {val: np.mean(accs) for val, accs in grouped_accuracies.items()}
        
        # 按参数值排序
        sorted_items = sorted(avg_accuracies.items())
        sorted_values = [item[0] for item in sorted_items]
        sorted_accuracies = [item[1] for item in sorted_items]
        
        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_values, sorted_accuracies, 'o-', linewidth=2)
        plt.title(f'{param_of_interest}对验证准确率的影响')
        plt.xlabel(param_of_interest)
        plt.ylabel('平均验证准确率')
        plt.grid(True)
        
        if isinstance(sorted_values[0], (int, float)):
            if param_of_interest == 'learning_rate' or param_of_interest == 'lambda_reg':
                plt.xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def visualize_parameter_interactions(self, param1, param2, save_path=None):
        """
        可视化两个超参数之间的交互
        
        参数:
            param1: 第一个参数名称
            param2: 第二个参数名称
            save_path: 保存图表的路径
        """
        if not self.results:
            print("没有可视化的结果，请先运行grid_search")
            return
        
        # 提取参数组合和对应的准确率
        param1_values = [r[param1] for r in self.results]
        param2_values = [r[param2] for r in self.results]
        accuracies = [r['best_val_accuracy'] for r in self.results]
        
        # 获取唯一参数值
        unique_param1 = sorted(set(param1_values))
        unique_param2 = sorted(set(param2_values))
        
        # 创建热力图数据
        heatmap_data = np.zeros((len(unique_param1), len(unique_param2)))
        count_data = np.zeros((len(unique_param1), len(unique_param2)))
        
        # 填充热力图数据
        for p1, p2, acc in zip(param1_values, param2_values, accuracies):
            i = unique_param1.index(p1)
            j = unique_param2.index(p2)
            heatmap_data[i, j] += acc
            count_data[i, j] += 1
        
        # 计算平均准确率
        mask = count_data > 0
        heatmap_data[mask] = heatmap_data[mask] / count_data[mask]
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, interpolation='nearest', cmap='viridis')
        plt.colorbar(label='验证准确率')
        
        # 设置坐标轴
        plt.xticks(np.arange(len(unique_param2)), unique_param2, rotation=45)
        plt.yticks(np.arange(len(unique_param1)), unique_param1)
        
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.title(f'{param1}和{param2}对验证准确率的交互影响')
        
        # 显示数值
        for i in range(len(unique_param1)):
            for j in range(len(unique_param2)):
                if count_data[i, j] > 0:
                    plt.text(j, i, f"{heatmap_data[i, j]:.3f}", 
                             ha="center", va="center", 
                             color="white" if heatmap_data[i, j] > 0.6 else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

def get_default_param_grid():
    """返回默认的超参数网格"""
    return {
        'hidden_size': [64, 128, 256, 512],
        'activation': ['relu', 'sigmoid', 'tanh'],
        'learning_rate': [0.01, 0.005, 0.001],
        'lr_decay': [1.0, 0.95, 0.9],
        'lambda_reg': [0.0, 0.0001, 0.001, 0.01],
        'batch_size': [32, 64, 128],
        'epochs': [10]  # 为了快速搜索，使用较少的epoch数
    } 