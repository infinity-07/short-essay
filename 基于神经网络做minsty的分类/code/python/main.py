import mnist_loader
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
for x,y in training_data:
    np.savetxt('train_x.csv',x, delimiter = ',')
    np.savetxt('train_y.csv',y, delimiter = ',')
    break

import network
net = network.Network([784, 30, 10])

net.SGD(training_data,1,10,3.0,test_data=test_data)
