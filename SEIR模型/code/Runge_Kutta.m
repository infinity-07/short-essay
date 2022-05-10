function  y = Runge_Kutta(dif_f,y0,x)
h = x(2)-x(1);
y = zeros(length(y0),length(x));               % 预分配内存
y(:,1) = y0;                    % 设置初始值
for i = 1:length(x)-1
    
    K_1 = dif_f(x(i),y(:,i));
    K_2 = dif_f(x(i)+h/2,y(:,i)+h/2*K_1);
    K_3 = dif_f(x(i)+h/2,y(:,i)+h/2*K_2);
    K_4 = dif_f(x(i)+h,y(:,i)+h*K_3);
    delta = (K_1+2*K_2+2*K_3+K_4)/6;
    
    y(:,i+1) = y(:,i) + delta*h;
end

end