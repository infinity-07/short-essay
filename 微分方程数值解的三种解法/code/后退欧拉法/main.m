clc,clear,close all
syms x y
dif_f = @(x,y) -20*x;
n = 100;
h = 1/n;
a = 0;
b = 1;
y0 = 1;
x = a:h:b;

y = euler_back(dif_f,y0,x);
plot(x,y,'LineWidth',2)
hold on