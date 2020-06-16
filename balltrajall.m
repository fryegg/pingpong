function tra = balltrajall(cam)
% bxo = camx;
% byo = camy;                       %%%ī�޶� �� �޾ƿ��� �ڵ�
% bzo = camz;
% bx1 = camx;
% by1 = camy;
% bz1 = camz;
% t=
% cam=strsplit(u,',');
bxo=cam(1);byo=cam(2);bzo=cam(3);t0=cam(4);bx1=cam(5);by1=cam(6);bz1=cam(7);t1=cam(8);
vxo = (bx1-bxo)./(t1-t0);
vyo = (by1-byo)./(t1-t0);        %%%�ӵ����ϱ�
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
ut = 0.102;             %�������
et = 0.883;             %ź�����
g = 9.802;              %�߷°��ӵ�


%bx = vxo.*(1-exp(-c*v*t))./(c*v)+bxo;                       % position x function
%by = vyo.*(1-exp(-c*v*t))./(c*v)+byo;                       % position y function
%bz = bzo-(((1./(c*v))*t)+(1./(c*v*c*v)).*exp(-c*v*t)-(1./(c*c*v*v))).*g+((1./(c*v)).*(1-exp(-c*v*t)).*vzo); % position z function

syms t;                             % z �� 0�϶� �ð� ���ϴ� �Լ�
equation = bzo-(((1./(c*v))*t)+(1./(c*v*c*v)).*exp(-c*v*t)-(1./(c*c*v*v))).*g+((1./(c*v)).*(1-exp(-c*v*t)).*vzo) == 0;      
solx = solve(equation);

bxland = vxo.*(1-exp(-c*v*solx))./(c*v)+bxo;     %Ź���뿡 ����� �� x ��ǥ
byland = vyo.*(1-exp(-c*v*solx))./(c*v)+byo;     %Ź���뿡 ����� �� y ��ǥ
%bzland = 0; %Ź���뿡 ����� �� z ��ǥ

vxland = vxo*exp(-c*v*solx);    %Ź���뿡 ����� �� x �ӵ�
vyland = vyo*exp(-c*v*solx);    %Ź���뿡 ����� �� y �ӵ�
vzland = (vzo+(g./(c*v)))*exp(-c*v*solx)-(g./(c*v));%Ź���뿡 ����� �� z �ӵ�

velres = ut.*(1+et).*((vzland)./(sqrt((vxland-rb*wy).^2+(vyland+rb*wx).^2))); %sliding parameter

vxout = (1-velres).*vxland+velres*rb*wy; %ƨ�ܳ����� x �ӵ�
vyout = (1-velres).*vyland-velres*rb*wx; %ƨ�ܳ����� y balltrajall([-1.06599,0.0496914,1.542,0.1512,0.143173,0.177274,1.629,0.2000])�ӵ�
vzout = -et*vzland;                      %ƨ�ܳ����� z �ӵ�

vaf = sqrt(vxout.^2+vyout.^2+vzout.^2);  %ƨ�� ���� �ӵ�(scalar)

syms t2;
equation2 = vxout.*(1-exp(-c*vaf*t2))./(c*vaf)+bxland == 2.6;  % x = 2.6m�� �� �ð� ���ϱ�
solx2 = solve(equation2);

%bxp = vxout.*(1-exp(-c*vaf*t2))./(c*vaf)+bxland;                    % ƨ�� ���� ���� x ����
%byp = vyout.*(1-exp(-c*vaf*t2))./(c*vaf)+byland;                    % ƨ�� ���� ���� y ����
%bzp = -(((1./(c*vaf))*t2)+(1./(c*vaf*c*vaf)).*exp(-c*vaf*t2)-(1./(c*c*vaf*vaf))).*g+((1./(c*vaf)).*(1-exp(-c*vaf*t2)).*vzout);  % ƨ�� ���� ���� z ����

bxf = vxout.*(1-exp(-c*vaf*solx2))./(c*vaf)+bxland;                 % ƨ�� ���� ���� x ��ġ
byf = vyout.*(1-exp(-c*vaf*solx2))./(c*vaf)+byland;                 % ƨ�� ���� ���� y ��ġ
bzf = -(((1./(c*vaf))*solx2)+(1./(c*vaf*c*vaf)).*exp(-c*vaf*solx2)-(1./(c*c*vaf*vaf))).*g+((1./(c*vaf)).*(1-exp(-c*vaf*solx2)).*vzout); % ƨ�� ���� ���� z ��ġ

vxf = vxout*exp(-c*vaf*solx2);   % x = 2.6m �� �� x �ӵ�
vyf = vyout*exp(-c*vaf*solx2);   % x = 2.6m �� �� y �ӵ�
vzf = (vzout+(g./(c*vaf)))*exp(-c*vaf*solx2)-(g./(c*vaf));% x = 2.6m �� �� z �ӵ�
time_total = solx+solx2;

tra =[double(bxf),double(byf),double(bzf),double(vxf),double(vyf),double(vzf),double(time_total)];
end