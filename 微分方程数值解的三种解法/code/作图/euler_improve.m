function y = euler_improve(dif_f,y0,x)
h = x(2)-x(1);
y = zeros(1,length(x));
y(1) = y0;
for i = 1:length(x)-1
    temp = y(i) + h*dif_f(x(i),y(i));
    delta = (dif_f(x(i),y(i))+dif_f(x(i+1),temp))/2;
    y(i+1) = y(i)+delta*h;
end
end