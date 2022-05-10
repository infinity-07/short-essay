clc,clear,close all
a = 0;
b = pi;
n = 10;
h = (b-a)/n;
x = a:h:b;
A1 = diag(-ones(1,n-1).*(x(2:end-1))/h^2);
A2 = diag(ones(1,n-1).*(1+2*x(2:end-1)/h^2));
A3 = A1;
Atop = [1,zeros(1,n)];
Amid = [A1,zeros(n-1,2)] + [zeros(n-1,1),A2,zeros(n-1,1)] + [zeros(n-1,2),A3];
Abottom = [zeros(1,n),1];

A = [Atop;Amid;Abottom];
b = [1,(x(2:end-1)+1).*cos(x(2:end-1)),-1]';
u = A\b;
plot(x,u,'linewidth',2)
hold on
x = a:0.1:pi;
plot(x,cos(x),'*')
legend('数值解','解析解')
