clc,clear,close all
n = 100;
syms x y
dif_f = @(x,y) -20*x;
h = 1/n;
a = 0;
b = 1;
y0 = 3;
x = a:h:b;

y = euler_improve(dif_f,y0,x);
plot(x,y,'LineWidth',2)