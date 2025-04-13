import numpy as np
import argparse
import os

from model import ThreeLayerNN
from train import Trainer
from test import test_model, load_and_test, evaluate_detailed
from utils import load_cifar10_data, plot_loss_and_accuracy, visualize_weights
from hyperparameter_tuning import HyperparameterTuner, get_default_param_grid

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='三层神经网络分类器')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'tune'],
                       help='运行模式: 训练(train), 测试(test), 超参数调优(tune)')
    
    # 数据相关
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--val_size', type=int, default=5000,
                       help='验证集大小')
    
    # 模型相关
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='隐藏层神经元数量')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'],
                       help='激活函数类型')
    
    # 训练相关
    parser.add_argument('--learning_rate', type=float, default=0.005,
                       help='学习率')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                       help='学习率衰减系数')
    parser.add_argument('--lambda_reg', type=float, default=0.0005,
                       help='L2正则化系数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    
    # 超参数调优相关
    parser.add_argument('--max_evals', type=int, default=30,
                       help='最大评估次数（用于超参数调优）')
    
    # 测试相关
    parser.add_argument('--model_path', type=str, default='models/best_model.npz',
                       help='测试模式下加载的模型路径')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print("加载CIFAR-10数据集...")
    train_X, train_y, val_X, val_y, test_X, test_y = load_cifar10_data(
        data_dir=args.data_dir,
        normalize=True,
        flatten=True,
        validation_size=args.val_size
    )
    
    # 输入、输出维度
    input_size = train_X.shape[1]  # 3072 for CIFAR-10 (32x32x3)
    output_size = 10  # CIFAR-10有10个类别
    
    # 训练模式
    if args.mode == 'train':
        print(f"\n开始训练，隐藏层大小: {args.hidden_size}, 激活函数: {args.activation}")
        
        # 创建模型
        model = ThreeLayerNN(input_size, args.hidden_size, output_size, args.activation)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            learning_rate=args.learning_rate,
            lr_decay=args.lr_decay,
            lambda_reg=args.lambda_reg
        )
        
        # 训练模型
        history = trainer.train(
            train_X, train_y, val_X, val_y, 
            batch_size=args.batch_size, 
            epochs=args.epochs
        )
        
        # 绘制训练历史
        trainer.plot_training_history(save_path='training_history.png')
        
        # 可视化模型权重
        visualize_weights(model, save_path='weight_visualization.png')
        
        # 在测试集上评估模型
        test_accuracy = test_model(model, test_X, test_y)
        print(f"\n测试集准确率: {test_accuracy:.4f}")
        
    # 测试模式
    elif args.mode == 'test':
        print(f"\n加载模型 {args.model_path} 并在测试集上评估")
        
        # 加载模型并评估
        results = load_and_test(
            model_path=args.model_path,
            X_test=test_X,
            y_test=test_y,
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=output_size,
            activation=args.activation
        )
    
    # 超参数调优模式
    elif args.mode == 'tune':
        print("\n开始超参数调优")
        
        # 创建超参数调优器
        tuner = HyperparameterTuner(
            X_train=train_X,
            y_train=train_y,
            X_val=val_X,
            y_val=val_y,
            input_size=input_size,
            output_size=output_size
        )
        
        # 获取默认参数网格
        param_grid = get_default_param_grid()
        
        # 运行网格搜索
        best_params, all_results = tuner.grid_search(
            param_grid=param_grid,
            max_evals=args.max_evals
        )
        
        # 可视化每个超参数的影响
        for param_name in param_grid.keys():
            if len(param_grid[param_name]) > 1:  # 只可视化有多个值的参数
                tuner.visualize_results(
                    param_of_interest=param_name,
                    save_path=f'tuning_results/{param_name}_impact.png'
                )
        
        # 可视化关键超参数之间的相互作用
        # 只在参数组合足够丰富时进行
        if len(all_results) > 5:
            key_params = ['hidden_size', 'learning_rate', 'lambda_reg']
            for i, param1 in enumerate(key_params):
                for param2 in key_params[i+1:]:
                    if len(param_grid[param1]) > 1 and len(param_grid[param2]) > 1:
                        tuner.visualize_parameter_interactions(
                            param1=param1,
                            param2=param2,
                            save_path=f'tuning_results/{param1}_{param2}_interaction.png'
                        )
        
        # 使用最佳参数训练模型
        print("\n使用最佳参数训练最终模型...")
        
        best_model = ThreeLayerNN(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            output_size=output_size,
            activation=best_params['activation']
        )
        
        best_trainer = Trainer(
            model=best_model,
            learning_rate=best_params['learning_rate'],
            lr_decay=best_params['lr_decay'],
            lambda_reg=best_params['lambda_reg']
        )
        
        # 最终训练使用更多轮数
        final_epochs = best_params['epochs'] * 2
        
        # 训练最终模型
        best_trainer.train(
            train_X, train_y, val_X, val_y,
            batch_size=best_params['batch_size'],
            epochs=final_epochs
        )
        
        # 绘制训练历史
        best_trainer.plot_training_history(save_path='best_model_training_history.png')
        
        # 可视化最终模型权重
        visualize_weights(best_model, save_path='best_model_weights.png')
        
        # 在测试集上评估最终模型
        test_accuracy = test_model(best_model, test_X, test_y)
        print(f"\n最终模型在测试集上的准确率: {test_accuracy:.4f}")
        
        # 保存最终模型
        best_model.save_weights('models/final_tuned_model.npz')
        print("最终模型已保存为 'models/final_tuned_model.npz'")

if __name__ == '__main__':
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    main() 