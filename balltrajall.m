function tra = balltrajall(cam)
% bxo = camx;
% byo = camy;                       %%%카메라 값 받아오는 코드
% bzo = camz;
% bx1 = camx;
% by1 = camy;
% bz1 = camz;
% t=
% cam=strsplit(u,',');
bxo=cam(1);byo=cam(2);bzo=cam(3);t0=cam(4);bx1=cam(5);by1=cam(6);bz1=cam(7);t1=cam(8);
vxo = (bx1-bxo)./(t1-t0);
vyo = (by1-byo)./(t1-t0);        %%%속도구하기
vzo = (bz1-bzo)./(t1-t0);

c = 0.1411;             %Drag coefficient    
m = 0.0025;             %mass
t = 0:0.01:4;           %time    
rb = 0.02;              %ball radius
% vxo = 4;                %Initial x velocity
% vyo = 0;                %Initial x velocity
% vzo = 1;                %Initial x velocity
v = sqrt(vxo^2+vyo^2+vzo^2); %Initial velocity(scalar)
%bxo = 0;                %Initial x position
%byo = 0;                %Initial y position
%bzo = 0.4;              %Initial z position
wx = 0;               %Initial x angle velocity
wy = 0;                %Initial y angle velocity
wz = 0;                %Initial z angle velocity
ut = 0.102;             %마찰계수
et = 0.883;             %탄성계수
g = 9.802;              %중력가속도


%bx = vxo.*(1-exp(-c*v*t))./(c*v)+bxo;                       % position x function
%by = vyo.*(1-exp(-c*v*t))./(c*v)+byo;                       % position y function
%bz = bzo-(((1./(c*v))*t)+(1./(c*v*c*v)).*exp(-c*v*t)-(1./(c*c*v*v))).*g+((1./(c*v)).*(1-exp(-c*v*t)).*vzo); % position z function

syms t;                             % z 가 0일때 시간 구하는 함수
equation = bzo-(((1./(c*v))*t)+(1./(c*v*c*v)).*exp(-c*v*t)-(1./(c*c*v*v))).*g+((1./(c*v)).*(1-exp(-c*v*t)).*vzo) == 0;      
solx = solve(equation);

bxland = vxo.*(1-exp(-c*v*solx))./(c*v)+bxo;     %탁구대에 닿았을 때 x 좌표
byland = vyo.*(1-exp(-c*v*solx))./(c*v)+byo;     %탁구대에 닿았을 때 y 좌표
%bzland = 0; %탁구대에 닿았을 때 z 좌표

vxland = vxo*exp(-c*v*solx);    %탁구대에 닿았을 때 x 속도
vyland = vyo*exp(-c*v*solx);    %탁구대에 닿았을 때 y 속도
vzland = (vzo+(g./(c*v)))*exp(-c*v*solx)-(g./(c*v));%탁구대에 닿았을 때 z 속도

velres = ut.*(1+et).*((vzland)./(sqrt((vxland-rb*wy).^2+(vyland+rb*wx).^2))); %sliding parameter

vxout = (1-velres).*vxland+velres*rb*wy; %튕겨나가는 x 속도
vyout = (1-velres).*vyland-velres*rb*wx; %튕겨나가는 y balltrajall([-1.06599,0.0496914,1.542,0.1512,0.143173,0.177274,1.629,0.2000])속도
vzout = -et*vzland;                      %튕겨나가는 z 속도

vaf = sqrt(vxout.^2+vyout.^2+vzout.^2);  %튕긴 뒤의 속도(scalar)

syms t2;
equation2 = vxout.*(1-exp(-c*vaf*t2))./(c*vaf)+bxland == 2.6;  % x = 2.6m일 때 시간 구하기
solx2 = solve(equation2);

%bxp = vxout.*(1-exp(-c*vaf*t2))./(c*vaf)+bxland;                    % 튕긴 뒤의 공의 x 궤적
%byp = vyout.*(1-exp(-c*vaf*t2))./(c*vaf)+byland;                    % 튕긴 뒤의 공의 y 궤적
%bzp = -(((1./(c*vaf))*t2)+(1./(c*vaf*c*vaf)).*exp(-c*vaf*t2)-(1./(c*c*vaf*vaf))).*g+((1./(c*vaf)).*(1-exp(-c*vaf*t2)).*vzout);  % 튕긴 뒤의 공의 z 궤적

bxf = vxout.*(1-exp(-c*vaf*solx2))./(c*vaf)+bxland;                 % 튕긴 뒤의 공의 x 위치
byf = vyout.*(1-exp(-c*vaf*solx2))./(c*vaf)+byland;                 % 튕긴 뒤의 공의 y 위치
bzf = -(((1./(c*vaf))*solx2)+(1./(c*vaf*c*vaf)).*exp(-c*vaf*solx2)-(1./(c*c*vaf*vaf))).*g+((1./(c*vaf)).*(1-exp(-c*vaf*solx2)).*vzout); % 튕긴 뒤의 공의 z 위치

vxf = vxout*exp(-c*vaf*solx2);   % x = 2.6m 일 때 x 속도
vyf = vyout*exp(-c*vaf*solx2);   % x = 2.6m 일 때 y 속도
vzf = (vzout+(g./(c*vaf)))*exp(-c*vaf*solx2)-(g./(c*vaf));% x = 2.6m 일 때 z 속도
time_total = solx+solx2;

tra =[double(bxf),double(byf),double(bzf),double(vxf),double(vyf),double(vzf),double(time_total)];
end