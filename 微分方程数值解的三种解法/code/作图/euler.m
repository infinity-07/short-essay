function  y = euler(f,a,b,h,y0)
x = a:h:b;                      % x_i
n = length(x)-1;                % n
y = zeros(1,n+1);               % 预分配内存
y(1) = y0;                     % 设置初始值
for i = 1:length(x)-1
    delta = f(x(i),y(i));
    y(i+1) = y(i)+delta*h;
end

end