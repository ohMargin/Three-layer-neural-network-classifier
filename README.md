# 三层神经网络分类器 (Three-layer Neural Network Classifier)

一个从零开始实现的三层神经网络分类器，用于CIFAR-10数据集的图像分类。该实现不依赖任何深度学习框架，仅使用NumPy手动实现反向传播和数据增强。

## 功能特点

- 完全使用NumPy从零实现神经网络
- 手动实现反向传播算法
- 支持多种激活函数（ReLU、Sigmoid、Tanh）
- 实现SGD优化器和学习率下降
- 实现交叉熵损失函数和L2正则化
- 支持多种高级数据增强技术
- 支持超参数调优
- 可视化训练过程和样本图像

## 项目结构

- `model.py`: 包含神经网络模型定义
- `train.py`: 训练模型的代码
- `test.py`: 测试模型性能的代码
- `hyperparameter_tuning.py`: 超参数搜索代码
- `utils.py`: 工具函数和数据增强
- `main.py`: 主程序入口
- `demo.py`: 参数化演示脚本

## 安装

1. 克隆仓库：
```
git clone https://github.com/yourusername/three-layer-neural-network-classifier.git
cd three-layer-neural-network-classifier
```

2. 安装依赖：
```
pip install numpy matplotlib seaborn scipy
```

## 使用方法

### 快速演示

运行演示脚本，快速体验模型训练和评估过程：

```
python demo.py
```

这将在CIFAR-10数据集上训练模型，并生成可视化结果。

### 使用数据增强

使用数据增强训练模型，提高泛化能力：

```
python demo.py --data_augmentation --horizontal_flip --random_erasing --random_crop
```

可用的数据增强选项：
- `--horizontal_flip`: 水平翻转图像
- `--vertical_flip`: 垂直翻转图像
- `--random_crop`: 随机裁剪图像
- `--random_erasing`: 随机擦除图像区域
- `--gaussian_noise`: 添加高斯噪声
- `--color_jitter`: 颜色抖动
- `--brightness_range`: 亮度调整范围（0-1）

### 调整模型参数

调整各种参数以达到最佳性能：

```
python demo.py --hidden_size 512 --learning_rate 0.005 --lr_decay 0.95 --lambda_reg 0.0005 --batch_size 128 --epochs 60
```

参数说明：
- `--hidden_size`: 隐藏层神经元数量 (推荐: 128, 256, 512, 1024)
- `--activation`: 激活函数类型 (relu, sigmoid, tanh)
- `--learning_rate`: 学习率 (推荐: 0.01, 0.005, 0.001)
- `--lr_decay`: 学习率衰减系数 (推荐: 1.0, 0.98, 0.95, 0.9)
- `--lambda_reg`: L2正则化系数 (推荐: 0.001, 0.0005, 0.0001, 0)
- `--batch_size`: 批次大小 (推荐: 32, 64, 128, 256)
- `--epochs`: 训练轮数 (推荐: 20, 30, 50)

### 可视化数据集样本

使用以下选项可视化数据集样本：

```
python demo.py --sample_viz
```

### 复杂训练示例

以下是一个结合数据增强和优化参数的完整示例：

```
python demo.py --hidden_size 512 --learning_rate 0.005 --lr_decay 0.95 --lambda_reg 0.0005 --batch_size 128 --epochs 60 --data_augmentation --horizontal_flip --random_erasing --random_crop --sample_viz
```

### 使用main.py进行训练

使用以下命令进行完整训练：

```
python main.py --mode train --hidden_size 512 --activation relu --learning_rate 0.005 --epochs 30
```

### 测试模型

使用以下命令测试已训练的模型：

```
python main.py --mode test --model_path models/best_model.npz
```

### 超参数调优

使用以下命令进行超参数搜索：

```
python main.py --mode tune --max_evals 30
```

这将自动搜索最佳超参数组合，并使用找到的最佳参数训练最终模型。

## 模型架构

该实现是一个具有一个隐藏层的三层神经网络：

1. 输入层：3072个神经元（对应CIFAR-10的32x32x3像素）
2. 隐藏层：可配置大小（默认200个神经元，可提高到512或1024获得更好性能）
3. 输出层：10个神经元（对应CIFAR-10的10个类别）

支持的激活函数：
- ReLU: max(0, x)
- Sigmoid: 1 / (1 + exp(-x))
- Tanh: tanh(x)

输出层使用Softmax激活函数，损失函数为交叉熵。

## 数据增强

本项目实现了多种数据增强技术，能够在每个训练批次应用随机变换：

1. **水平翻转**：随机水平翻转图像
2. **垂直翻转**：随机垂直翻转图像
3. **随机裁剪**：随机裁剪图像并调整回原始大小
4. **随机擦除**：随机擦除图像的一部分区域
5. **高斯噪声**：添加随机高斯噪声
6. **颜色抖动**：调整图像的色相、饱和度和亮度
7. **亮度调整**：随机调整图像亮度

数据增强可显著提高模型的泛化能力和准确率。

## 实验结果

在CIFAR-10数据集上，使用默认配置（200个隐藏层神经元，ReLU激活函数），模型能够达到约45%的测试准确率。通过优化参数和使用数据增强，可以进一步提高性能：

- 隐藏层大小512 + 数据增强：约45-50%准确率
- 隐藏层大小1024 + 数据增强 + 更长训练时间：约48-55%准确率

每次实验的结果都会保存在`models/[experiment_name]`目录下，包括：
- `results.txt`：实验参数和结果摘要
- `training_history.png`：训练和验证损失曲线以及验证准确率曲线
- `sample_images.png`：数据集样本可视化（如果启用）
- `final_model.npz`：训练后的模型权重

## 优化建议

为了获得最佳性能，建议：

1. 增加隐藏层大小（512或1024）
2. 使用水平翻转、随机裁剪和随机擦除数据增强
3. 设置较小的学习率（0.005）和适当的衰减（0.95）
4. 使用较小的L2正则化系数（0.0005）
5. 训练更长时间（50-100轮）