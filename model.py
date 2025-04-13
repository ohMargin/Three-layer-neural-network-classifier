import numpy as np

class ActivationFunction:
    """激活函数类，支持多种激活函数"""
    
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return (x > 0).astype(float)
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x, derivative=False):
        s = 1 / (1 + np.exp(-x))
        if derivative:
            return s * (1 - s)
        return s
    
    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)
    
    @staticmethod
    def softmax(x, derivative=False):
        # Softmax函数实现
        # 为防止数值溢出，对每个样本减去最大值
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        if derivative:
            # 在反向传播中，softmax导数与交叉熵损失函数结合后会简化
            # 这里返回None，实际的梯度计算在损失函数中
            return None
        return softmax_x

class ThreeLayerNN:
    """三层神经网络模型"""
    
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """
        初始化神经网络
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层神经元数量
            output_size: 输出类别数量
            activation: 激活函数类型('relu', 'sigmoid', 'tanh')
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = ActivationFunction.relu
        elif activation == 'sigmoid':
            self.activation = ActivationFunction.sigmoid
        elif activation == 'tanh':
            self.activation = ActivationFunction.tanh
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出层始终使用softmax激活
        self.output_activation = ActivationFunction.softmax
        
        # He初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # 缓存用于反向传播
        self.cache = {}
        
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据，形状为(batch_size, input_size)
            
        返回:
            输出预测结果
        """
        # 第一层：线性变换 + 激活函数
        z1 = X.dot(self.W1) + self.b1
        a1 = self.activation(z1)
        
        # 第二层（输出层）：线性变换 + softmax激活
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.output_activation(z2)
        
        # 保存中间值以用于反向传播
        self.cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }
        
        return a2
    
    def backward(self, y, output_grad=None, lambda_reg=0.0):
        """
        反向传播计算梯度
        
        参数:
            y: 真实标签，形状为(batch_size, output_size)，如果是one-hot编码
               或形状为(batch_size,)，如果是类别索引
            output_grad: 输出层梯度（如果不是None）
            lambda_reg: L2正则化强度
            
        返回:
            梯度字典
        """
        batch_size = self.cache['X'].shape[0]
        
        # 如果y不是one-hot编码，转换为one-hot编码
        if len(y.shape) == 1:
            y_one_hot = np.zeros((batch_size, self.output_size))
            y_one_hot[np.arange(batch_size), y] = 1
            y = y_one_hot
        
        # 获取缓存的前向传播值
        X = self.cache['X']
        a1 = self.cache['a1']
        a2 = self.cache['a2']
        
        # 输出层梯度：对于交叉熵损失+softmax
        if output_grad is None:
            # 如果未提供输出梯度，则假设使用交叉熵损失
            # 交叉熵损失对softmax的导数简化为: a2 - y
            dz2 = a2 - y
        else:
            dz2 = output_grad
        
        # 计算W2和b2的梯度
        dW2 = a1.T.dot(dz2) / batch_size + 2 * lambda_reg * self.W2  # 加上L2正则项
        db2 = np.sum(dz2, axis=0) / batch_size
        
        # 计算隐藏层梯度
        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * self.activation(self.cache['z1'], derivative=True)
        
        # 计算W1和b1的梯度
        dW1 = X.T.dot(dz1) / batch_size + 2 * lambda_reg * self.W1  # 加上L2正则项
        db1 = np.sum(dz1, axis=0) / batch_size
        
        # 返回所有梯度
        return {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }
    
    def save_weights(self, filename):
        """保存模型权重"""
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    
    def load_weights(self, filename):
        """加载模型权重"""
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
    
    def predict(self, X):
        """预测类别"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1) 