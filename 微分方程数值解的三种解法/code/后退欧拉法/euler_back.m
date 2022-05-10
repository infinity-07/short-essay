function y = euler_back(dif_f,y0,x)
h = x(2)-x(1);
y = zeros(1,length(x));
y(1) = y0;

k = 10;

for i = 1:length(x)-1
    yy = y(i) + dif_f(x(i+1),y(i))*h;
    for j = 1:k
        yy = y(i) + dif_f(x(i+1),yy)*h;
    end
    y(i+1) = y(i) + dif_f(x(i+1),yy)*h;
end
end