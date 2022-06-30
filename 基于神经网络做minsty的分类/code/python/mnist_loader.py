# %load mnist_loader.py

# 加载 MNIST 图像数据的库。 有关返回的数据结构的详细信息，请参阅“load_data”和“load_data_wrapper”的文档字符串。 
# 在实践中，``load_data_wrapper`` 是我们的神经网络代码通常调用的函数。


#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    #将 MNIST 数据作为包含训练数据、验证数据和测试数据的元组返回。 
    # ``training_data`` 作为包含两个条目的元组返回。 
    # 第一个条目包含实际的训练图像。 这是一个包含 50,000 个条目的 numpy ndarray。 
    # 每个条目依次是具有 784 个值的 numpy ndarray，表示单个 MNIST 图像中的 28 * 28 = 784 个像素。 
    # ``training_data`` 元组中的第二个条目是一个包含 50,000 个条目的 numpy ndarray。 
    # 这些条目只是元组第一个条目中包含的相应图像的数字值 (0...9)。
    #  ``validation_data`` 和 ``test_data`` 是相似的，除了每个只包含 10,000 个图像。 
    # 这是一种很好的数据格式，但对于在神经网络中使用，稍微修改 ``training_data`` 的格式会很有帮助。 
    # 这是在包装函数“load_data_wrapper()”中完成的，见下文。
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    # 返回一个包含``（training_data，validation_data，test_data）``的元组。 
    # 基于 ``load_data``，但该格式更方便在我们的神经网络实现中使用。 
    # 特别是，``training_data`` 是一个包含 50,000 个 2 元组 ``(x, y)`` 的列表。
    #  ``x`` 是一个包含输入图像的 784 维 numpy.ndarray。 
    # ``y`` 是一个 10 维的 numpy.ndarray，表示对应于``x`` 的正确数字的单位向量。 
    # ``validation_data`` 和 ``test_data`` 是包含 10,000 个 2 元组 ``(x, y)`` 的列表。 
    # 在每种情况下，“x”是包含输入图像的 784 维 umpy.ndarry，“y”是相应的分类，即对应于“x”的数字值（整数）。 
    # 显然，这意味着我们对训练数据和验证/测试数据使用的格式略有不同。 
    # 这些格式在我们的神经网络代码中被证明是最方便使用的。
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    # 返回一个 10 维单位向量，第 j 个位置为 1.0，其他位置为零。 
    # 这用于将数字 (0...9) 转换为来自神经网络的相应所需输出。
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
