# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

用于实现前馈神经网络的随机梯度下降学习算法的模块。 使用反向传播计算梯度。 请注意，我专注于使代码简单、易于阅读和易于修改。 它没有经过优化，并且省略了许多理想的功能。

"""

#### Libraries
# Standard library
import random
import csv
# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        # 列表“大小”包含网络各层中的神经元数量。 
        # 例如，如果列表是 [2, 3, 1]，那么它将是一个三层网络，第一层包含 2 个神经元，第二层包含 3 个神经元，第三层包含 1 个神经元。 
        # 网络的偏差和权重是随机初始化的，使用均值为 0，方差为 1 的高斯分布。
        # 请注意，第一层假定为输入层，按照惯例，我们不会为这些神经元设置任何偏差，因为偏差只用于计算后面层的输出。
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]             # 偏置
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]   # 权重

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size,eta,test_data=None):
        # training_data 是⼀个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出。
        # 变量 epochs 迭代期数量
        # mini_batch_size 采样时的⼩批量数据的⼤⼩。
        # eta 是学习速率，η。
        # 如果给出了可选参数 test_data，那么程序会在每个训练器后评估⽹络，并打印出部分进展。这对于追踪进度很有⽤，但相当拖慢执⾏速度。
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            # 把 training_data 分成 n/mini_batch_size 等分。每一份有 mini_batch_size 个元素
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]     
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

        np.savetxt('weight0.csv',self.weights[0], delimiter = ',')
        np.savetxt('weight1.csv',self.weights[1], delimiter = ',')
        np.savetxt('biases0.csv',self.biases[0], delimiter = ',')
        np.savetxt('biases1.csv',self.biases[1], delimiter = ',')


    def update_mini_batch(self, mini_batch, eta):
        # 通过使用反向传播对单个小批量应用梯度下降来更新网络的权重和偏差。 
        # ``mini_batch`` 是元组 ``(x, y)`` 的列表，而 ``eta`` 是学习率。
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        print(len(mini_batch))
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        # 返回一个元组 ``(nabla_b, nabla_w)`` 代表成本函数 C_x 的梯度。 
        # ``nabla_b`` 和 ``nabla_w`` 是 numpy 数组的逐层列表，类似于 ``self.biases`` 和 ``self.weights``。
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 请注意，下面循环中的变量 l 的使用与本书第 2 章中的符号略有不同。 
        # 这里，l = 1 表示最后一层神经元，l = 2 表示倒数第二层，以此类推。 
        # 这是对书中方案的重新编号，此处用于利用 Python 可以在列表中使用负索引的事实。
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        # 返回神经网络输出正确结果的测试输入的数量。 
        # 请注意，假设神经网络的输出是最后一层中激活最高的神经元的索引。

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        # 返回输出激活的偏导数向量 \partial C_x /\partial a。
        return (output_activations-y)

#### 杂项函数
def sigmoid(z):
    """sigmoid 函数."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """sigmoid 函数的导数"""
    return sigmoid(z)*(1-sigmoid(z))
