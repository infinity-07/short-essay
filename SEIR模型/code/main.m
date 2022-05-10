clc,clear,close all
N = 10000;
r = 5;
beta = 0.03;
gamma = 0.1;
a = 0.1;

dif_f = @(x,y) [-r*beta*y(3)*y(1)/N;
    r*beta*y(3)*y(1)/N-a*y(2);
    a*y(2)-gamma*y(3);
    gamma*y(3)];
y0 = [9999;
    0;
    1;
    0];
x = 0:1:150;


y = Runge_Kutta(dif_f,y0,x);
plot(x,y(1,:),x,y(2,:),x,y(3,:),x,y(4,:),"LineWidth",2)

legend('易感者','暴露者','患病者','康复者')
print("正常",'-depsc')