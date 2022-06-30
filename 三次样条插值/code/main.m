clc,clear,close all
x = [45 75 105 135 165 225 255];
y = [20 60 60 20 -60 -100 20];
n = length(x);
u = 0;
v = 3;
m = get_m(x,y,u,v);
xx = x(1):0.01:x(end);
yy = my_elmit(x,y,m,xx);
plot(x,y,'*')
hold on
plot(xx,yy,'LineWidth',1.5)
h=legend('$(x_i,y_i)$','Three spline interpolation');
set(h,'Interpreter','latex','FontName','Times New Roman','FontSize',15,'FontWeight','normal')