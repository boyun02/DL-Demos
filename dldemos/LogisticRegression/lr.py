import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iter=100000,
                 fit_intercept=True, penalty='l2', C=1.0):
        """
        初始化逻辑回归模型

        参数：
        - learning_rate: 学习率 (默认0.01)
        - num_iter: 迭代次数 (默认100000)
        - fit_intercept: 是否添加偏置项 (默认True)
        - penalty: 正则化类型 ('l1', 'l2' 或 None) (默认'l2')
        - C: 正则化强度的倒数，值越小正则化越强 (默认1.0)
        """
        if penalty not in ['l1', 'l2', None]:
            raise ValueError("Invalid penalty type. Expected 'l1', 'l2' or None")

        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.C = C
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def __add_intercept(self, X):
        """添加偏置项"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-z))

    def __compute_regularization(self, theta):
        """计算正则化项"""
        if self.penalty is None or self.C <= 0:
            return 0

        alpha = 1 / self.C  # 正则化强度系数
        # 不惩罚偏置项 (theta[0])
        if self.penalty == 'l2':
            return 0.5 * alpha * np.sum(theta[1:] ** 2)
        elif self.penalty == 'l1':
            return alpha * np.sum(np.abs(theta[1:]))

    def __compute_gradient_penalty(self, theta):
        """计算正则化梯度"""
        if self.penalty is None or self.C <= 0:
            return np.zeros_like(theta)

        alpha = 1 / self.C
        gradient = np.zeros_like(theta)
        # 不惩罚偏置项 (theta[0])
        if self.penalty == 'l2':
            gradient[1:] = alpha * theta[1:]
        elif self.penalty == 'l1':
            gradient[1:] = alpha * np.sign(theta[1:])
        return gradient

    def __loss(self, h, y, theta):
        """计算带正则化的损失函数"""
        h = np.clip(h, 1e-8, 1 - 1e-8)  # 数值稳定处理
        # 交叉熵损失
        cross_entropy = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        # 正则化项
        reg_term = self.__compute_regularization(theta)
        return cross_entropy + reg_term

    def fit(self, X, y, X_val=None, y_val=None):
        """
        训练模型并记录训练过程指标

        参数：
        - X: 训练特征矩阵
        - y: 训练标签
        - X_val: 验证特征矩阵 (可选)
        - y_val: 验证标签 (可选)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        if self.fit_intercept:
            X = self.__add_intercept(X)
            # if X_val is not None:
            #     X_val = self.__add_intercept(X_val)

        # 初始化参数
        self.theta = np.zeros((X.shape[1], 1))

        # 清空历史记录
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        for i in range(self.num_iter):
            # 前向传播
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)

            # 计算梯度
            gradient = np.dot(X.T, (h - y)) / y.size
            gradient += self.__compute_gradient_penalty(self.theta)

            # 参数更新
            self.theta -= self.learning_rate * gradient

            # 每1000次记录指标
            if i % 1000 == 0:
                # 计算当前损失
                current_loss = self.__loss(h, y, self.theta)
                self.loss_history.append(current_loss)

                # 计算训练集准确率
                train_pred = (h >= 0.5).astype(int)
                train_acc = np.mean(train_pred == y)
                self.train_acc_history.append(train_acc)

                # 计算验证集准确率（如果提供）
                if X_val is not None and y_val is not None:
                    val_prob = self.predict_prob(X_val)
                    val_pred = (val_prob >= 0.5).astype(int)
                    val_acc = np.mean(val_pred == y_val.reshape(-1, 1))
                    self.val_acc_history.append(val_acc)

    def predict_prob(self, X):
        """预测概率"""
        X = np.array(X)  # 确保 X 是 NumPy 数组
        if self.fit_intercept:
            X = self.__add_intercept(X)  # 添加偏置项
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        """预测类别"""
        return (self.predict_prob(X) >= threshold).astype(int)


def plot_training_curve(models, metric='loss'):
    """
    绘制训练过程对比曲线

    参数：
    - models: 字典 {模型名称: 模型实例}
    - metric: 指标类型 ('loss', 'train_acc', 'val_acc')
    """
    plt.figure(figsize=(12, 6))

    for model_name, model in models.items():
        iterations = np.arange(len(getattr(model, f'{metric}_history'))) * 1000
        values = getattr(model, f'{metric}_history')
        plt.plot(iterations, values, label=model_name, linewidth=2)

    plt.title(f'Training {metric.capitalize()} Comparison', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# 示例使用 --------------------------------------------------
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    X = np.r_[np.random.randn(100, 2) + [2, 2],  # 正类
              np.random.randn(100, 2) - [2, 2]]  # 负类
    y = np.array([0] * 100 + [1] * 100)

    # 划分训练集和验证集
    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    # 初始化不同正则化策略的模型
    models = {
        'No Reg (C=1.0)': LogisticRegression(penalty=None, C=1.0, learning_rate=0.1),
        'L2 Reg (C=0.1)': LogisticRegression(penalty='l2', C=0.1, learning_rate=0.1),
        'L1 Reg (C=0.1)': LogisticRegression(penalty='l1', C=0.1, learning_rate=0.1)
    }

    # 训练所有模型
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train, X_val, y_val)
        print(f"{name} Final Parameters: {model.theta.ravel().round(2)}")
        print(f"Train Acc: {model.train_acc_history[-1]:.2%} | Val Acc: {model.val_acc_history[-1]:.2%}\n")

    # 可视化训练过程
    plot_training_curve(models, metric='loss')
    plot_training_curve(models, metric='train_acc')
    plot_training_curve(models, metric='val_acc')
