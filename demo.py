"""
演示脚本，展示模型训练和评估的基本流程
支持命令行参数，可以方便地尝试不同的超参数组合
"""
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
from model import ThreeLayerNN
from train import Trainer
from test import test_model
from utils import load_cifar10_data, get_cifar10_class_names, load_cifar10_data_with_advanced_augmentation, visualize_weights, visualize_model_parameters

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='三层神经网络分类器 - 参数实验')
    
    # 模型相关
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='隐藏层神经元数量 (推荐尝试: 128, 256, 512, 1024,2048)')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'],
                       help='激活函数类型')
    
    # 训练相关
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='初始学习率 (推荐尝试: 0.01, 0.005, 0.001)')
    parser.add_argument('--lr_decay', type=float, default=0.98,
                       help='学习率衰减系数 (推荐尝试: 1.0, 0.98, 0.95, 0.9)')
    parser.add_argument('--lambda_reg', type=float, default=0.001,
                       help='L2正则化系数 (推荐尝试: 0.001, 0.0005, 0.0001, 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小 (推荐尝试: 32, 64, 128, 256)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数 (推荐尝试: 20, 30, 50)')
    
    # 数据相关
    parser.add_argument('--val_size', type=int, default=5000,
                       help='验证集大小')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称，用于保存模型和结果')
    
    # 可视化相关
    parser.add_argument('--sample_viz', action='store_true',
                       help='是否可视化数据集样本')
    
    # 数据增强基本选项
    parser.add_argument('--data_augmentation', action='store_true',
                       help='是否启用数据增强')
    parser.add_argument('--horizontal_flip', action='store_true',
                       help='启用水平翻转')
    parser.add_argument('--vertical_flip', action='store_true',
                       help='启用垂直翻转')
    parser.add_argument('--brightness_range', type=float, default=0.0,
                       help='随机亮度调整范围')
    
    # 高级数据增强选项
    parser.add_argument('--random_crop', action='store_true',
                       help='启用随机裁剪')
    parser.add_argument('--color_jitter', action='store_true',
                       help='启用颜色抖动')
    parser.add_argument('--random_erasing', action='store_true',
                       help='启用随机擦除')
    parser.add_argument('--gaussian_noise', action='store_true',
                       help='启用高斯噪声')
    parser.add_argument('--mixup', action='store_true',
                       help='启用Mixup图像混合')
    parser.add_argument('--cutmix', action='store_true',
                       help='启用CutMix裁剪混合')
    
    return parser.parse_args()

def visualize_sample_images(X, y, class_names=None, n_samples=64, save_path=None):
    """
    可视化数据集中的样本图像
    
    参数:
        X: 图像数据，形状为(N, 3072)展平后的图像或(N, 32, 32, 3)的原始图像
        y: 标签，形状为(N,)
        class_names: 类别名称列表
        n_samples: 要可视化的样本数量
        save_path: 保存图表的路径
    """
    # 如果没有提供类别名称，使用CIFAR-10默认类别名称
    if class_names is None:
        class_names = get_cifar10_class_names()
    
    # 确保我们有足够的样本
    n_samples = min(n_samples, X.shape[0])
    
    # 选择随机样本
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    sample_X = X[indices]
    sample_y = y[indices]
    
    # 如果图像是展平的，重塑回原始形状
    if len(sample_X.shape) == 2:  # 如果是展平的 (N, 3072)
        sample_X = sample_X.reshape(-1, 32, 32, 3)
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    # 可视化样本
    plt.figure(figsize=(20, 20))
    for i in range(n_samples):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(sample_X[i])
        plt.title(class_names[sample_y[i]])
        plt.axis('off')
    
    plt.suptitle('CIFAR-10 样本图像', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为总标题留出空间
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
        print(f"样本图像已保存到 {save_path}")
    
    plt.show()

def demo(args=None):
    """运行演示"""
    # 如果没有提供命令行参数，使用默认值
    if args is None:
        args = parse_args()
    
    # 生成实验名称（如果未提供）
    if args.experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"exp_h{args.hidden_size}_lr{args.learning_rate}_reg{args.lambda_reg}_{timestamp}"
        # 如果使用了数据增强，在实验名称中标明
        if args.data_augmentation:
            args.experiment_name = f"aug_{args.experiment_name}"
    
    # 创建保存目录
    model_dir = os.path.join('models', args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"三层神经网络分类器 - 实验: {args.experiment_name}")
    print("=" * 80)
    
    print("\n当前参数配置:")
    print(f"- 隐藏层大小: {args.hidden_size}")
    print(f"- 激活函数: {args.activation}")
    print(f"- 学习率: {args.learning_rate}")
    print(f"- 学习率衰减: {args.lr_decay}")
    print(f"- L2正则化: {args.lambda_reg}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 训练轮数: {args.epochs}")
    print(f"- 验证集大小: {args.val_size}")
    
    # 如果启用了数据增强，设置增强参数
    augmentation_params = None
    if args.data_augmentation:
        print("\n启用数据增强:")
        augmentation_params = {
            # 基本参数
            'horizontal_flip': args.horizontal_flip,
            'vertical_flip': args.vertical_flip,
            'brightness_range': args.brightness_range,
            
            # 高级参数
            'random_crop': args.random_crop,
            'color_jitter': args.color_jitter,
            'random_erasing': args.random_erasing,
            'gaussian_noise': args.gaussian_noise,
            'mixup': args.mixup,
            'cutmix': args.cutmix
        }
    
    print("\n加载CIFAR-10数据集（这可能需要一些时间）...")
    
    # 是否需要可视化样本
    if args.sample_viz:
        # 加载不展平的数据用于可视化
        if args.data_augmentation:
            # 使用数据增强加载数据
            train_X_orig, train_y, val_X_orig, val_y, test_X_orig, test_y = load_cifar10_data_with_advanced_augmentation(
                normalize=True,
                flatten=False,  # 不展平，以便进行数据增强和可视化
                validation_size=args.val_size,
                augmentation_params=augmentation_params
            )
        else:
            # 不使用数据增强加载数据
            train_X_orig, train_y, val_X_orig, val_y, test_X_orig, test_y = load_cifar10_data(
                normalize=True,
                flatten=False,  # 不展平，以便进行可视化
                validation_size=args.val_size
            )
        
        # 可视化样本图像并保存
        print("\n可视化数据集样本...")
        sample_path = os.path.join(model_dir, "sample_images.png")
        visualize_sample_images(
            train_X_orig, 
            train_y, 
            n_samples=64,
            save_path=sample_path
        )
        
        # 展平数据用于训练
        train_X = train_X_orig.reshape(train_X_orig.shape[0], -1)
        val_X = val_X_orig.reshape(val_X_orig.shape[0], -1)
        test_X = test_X_orig.reshape(test_X_orig.shape[0], -1)
    else:
        # 直接加载展平的数据
        if args.data_augmentation:
            # 使用数据增强加载数据
            train_X, train_y, val_X, val_y, test_X, test_y = load_cifar10_data_with_advanced_augmentation(
                normalize=True,
                flatten=True,
                validation_size=args.val_size,
                augmentation_params=augmentation_params
            )
        else:
            # 不使用数据增强加载数据
            train_X, train_y, val_X, val_y, test_X, test_y = load_cifar10_data(
                normalize=True,
                flatten=True,
                validation_size=args.val_size
            )
    
    print(f"\n数据集大小:")
    print(f"- 训练集: {train_X.shape[0]} 样本")
    print(f"- 验证集: {val_X.shape[0]} 样本")
    print(f"- 测试集: {test_X.shape[0]} 样本")
    
    # 输入和输出维度
    input_size = train_X.shape[1]  # 3072 for CIFAR-10
    output_size = 10  # 10个类别
    
    print(f"\n创建三层神经网络:")
    print(f"- 输入层: {input_size} 神经元")
    print(f"- 隐藏层: {args.hidden_size} 神经元")
    print(f"- 输出层: {output_size} 神经元")
    
    # 创建模型
    model = ThreeLayerNN(input_size, args.hidden_size, output_size, args.activation)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        lambda_reg=args.lambda_reg
    )
    
    print(f"\n开始训练 {args.epochs} 轮:")
    start_time = time.time()
    
    # 使用完整训练集训练模型
    history = trainer.train(
        train_X, train_y, 
        val_X, val_y,
        batch_size=args.batch_size, 
        epochs=args.epochs,
        augmentation_params=augmentation_params  # 传递数据增强参数
    )
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
    print(f"最佳验证准确率: {trainer.best_val_accuracy:.4f}")
    
    # 在完整测试集上评估
    print("\n在完整测试集上评估模型...")
    test_start = time.time()
    test_accuracy = test_model(model, test_X, test_y)
    test_time = time.time() - test_start
    print(f"测试集准确率: {test_accuracy:.4f} (耗时: {test_time:.2f}秒)")
    
    # 保存实验结果
    results_file = os.path.join(model_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"实验名称: {args.experiment_name}\n")
        f.write(f"隐藏层大小: {args.hidden_size}\n")
        f.write(f"激活函数: {args.activation}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"学习率衰减: {args.lr_decay}\n")
        f.write(f"L2正则化: {args.lambda_reg}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"训练时间: {training_time:.2f}秒\n")
        f.write(f"最佳验证准确率: {trainer.best_val_accuracy:.4f}\n")
        f.write(f"测试集准确率: {test_accuracy:.4f}\n")
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model.npz")
    model.save_weights(final_model_path)
    
    # 绘制训练历史
    history_path = os.path.join(model_dir, "training_history.png")
    trainer.plot_training_history(save_path=history_path)
    
    # 注释掉或删除以下行
    # visualize_model_parameters(model, save_dir='model_analysis', use_chinese=False)
    
    print("\n实验完成!")
    print(f"结果已保存到 {model_dir} 目录")
    print(f"训练历史: {history_path}")
    
    return model, trainer, test_accuracy

if __name__ == "__main__":
    args = parse_args()
    demo(args) 