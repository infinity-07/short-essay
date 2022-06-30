clc,clear,close all

mnist_loader                % 读取数据
sizes = [784, 30, 10];      % 输入层，隐藏层，输出层神经元数量
epochs = 7;                 % 迭代次数
mini_batch_size = 10;       % 小样本数量
eta = 3;                    % 学习率     


%-------生成一个newwork---------%
biases = {randn(sizes(2),1),randn(sizes(3),1)}; % 偏置矩阵
weights = {randn(sizes(2),sizes(1)),randn(sizes(3),sizes(2))};  % 权重矩阵
[biases,weights] = net_SGD_train(sizes,biases,weights,train_x,train_y,test_x,test_y,epochs,mini_batch_size,3);

function  [biases,weights] = net_SGD_train(sizes,biases,weights,train_x,train_y,test_x,test_y,epochs,mini_batch_size,eta)
n = size(train_x,2);
for i = 1 : epochs
    randIndex_train = randperm(n);
    train_x = train_x(:,randIndex_train);
    train_y = train_y(:,randIndex_train);

    for j = 1:n/mini_batch_size
        mini_batchs_x = train_x(:,(j-1)*mini_batch_size+1:j*mini_batch_size);
        mini_batchs_y = train_y(:,(j-1)*mini_batch_size+1:j*mini_batch_size);
        [biases,weights] = update_mini_batch(mini_batchs_x,mini_batchs_y,eta,sizes,biases,weights);
    end
    correct_rate = evaluate(test_x,test_y,biases,weights);
    disp(strcat('第',num2str(i),'次迭代, 正确率为：',num2str(correct_rate),'%'))
end
end
%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%

function [biases,weights] = update_mini_batch(mini_batchs_x,mini_batchs_y, eta,sizes,biases,weights)

m = size(mini_batchs_x,2);
nabla_b = {zeros(sizes(2),1),zeros(sizes(3),1)};
nabla_w = {zeros(sizes(2),sizes(1)),zeros(sizes(3),sizes(2))};

for i = 1:m
    [delta_nabla_b,delta_nabla_w] = backprop(mini_batchs_x(:,i),mini_batchs_y(:,i),biases,weights);
    for j = 1:2
        nabla_b{j} = nabla_b{j} + delta_nabla_b{j};
        nabla_w{j} = nabla_w{j} + delta_nabla_w{j};
    end
end
for j = 1:2
    weights{j} = weights{j}-(eta/m)*nabla_w{j};
    biases{j} = biases{j}-(eta/m)*nabla_b{j};
end
end


%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%

function [nabla_b, nabla_w] = backprop(x, y,biases,weights)

nabla_b = {[],[]};
nabla_w = {[],[]};


activation = x;
activations = {0,0,0};
zs = {0,0};
activations{1} = x;
% activations = [x] # list to store all the activations, layer by layer
% zs = [] # list to store all the z vectors, layer by layer
for i = 1:2
    z = weights{i} * activation + biases{i};
    zs{i} = z;
    activation = sigmoid(z);
    activations{i+1} = activation;
end

delta = cost_derivative(activations{3},y) .* sigmoid_prime(zs{2});
nabla_b{2} = delta;
nabla_w{2} = delta * transpose(activations{2});


z = zs{1};
sp = sigmoid_prime(z);

delta = transpose(weights{2}) * delta .* sp;
nabla_b{1} = delta;
nabla_w{1} = delta * transpose(activations{1});


end



%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
function out = cost_derivative(output_activations, y)
out = output_activations - y;
end


%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%

function out = sigmoid(z)
out = 1./(1+exp(-z));
end

%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
function out = sigmoid_prime(z)
out = sigmoid(z).*(1-sigmoid(z));
end

%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
function a = feedforward(a,biases,weights)

for i = 1:2
    b = biases{i};
    w = weights{i};
    a = sigmoid(w*a+b);
end
end


%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%
%----------------------------------------------------------------------------%

function correct_rate = evaluate(test_x,test_y,biases,weights)
n = size(test_x,2);
correct = 0;
for i = 1:n
    y1 = feedforward(test_x(:,i),biases,weights);
    [~,y1] = max(y1);
    y2 = test_y(:,i);
    [~,y2] = max(y2);
    if y1==y2
        correct = correct + 1;
    end
end

correct_rate = correct/n*100;
end
