%% 
% $$y_{k+1}=y_k+hf(x_k,y_k)$$

clc,clear,close all
n = 100;
f = @(x,y) -20*x;             % 设置微分方程
a = 0;                          % 区间左端点
b = 1;                          % 区间右端点
h = 1/n;                        % 取点间隔
y0 = 1;                       % 设置初值

y = euler(f,a,b,h,y0);
plot(a:h:b,y)