clear all
close all
clc

system = TwoBodySystem;

system.p1 = Particle;
system.p2 = Particle;

system.p1.position_v = [0, 0, 0];
system.p2.position_v = [0, 1.5e11, 0];
system.p1.trajectory_v_a(1, :) = system.p1.position_v;
system.p2.trajectory_v_a(1, :) = system.p2.position_v;
system.p1.mass = 2e30;
system.p2.mass = 6e24;
%% Settings
% helix
system.p1.velocity_v = [-15.5e3, 0, 0];
system.p2.velocity_v = [15.5e3, 0, 15.5e3];

% opening helix
% system.p1.mass = 2e28;
% system.p2.mass = 2e28;
% system.p1.velocity_v = [-15.5e2, -15.5e2, 0];
% system.p2.velocity_v = [15.5e1, 0, 15.5e1];

% Dual helix
% system.p1.mass = 2e28;
% system.p2.mass = 2e28;
% system.p1.velocity_v = [-150e1, -150e1, 0];
% system.p2.velocity_v = [150e1, 0, 150e1];

% circular
% system.p1.velocity_v = [0, 0, 0];
% system.p2.velocity_v = [30e3, 0, 0];

% elliptical
% system.p1.velocity_v = [0, 0, 0];
% system.p2.velocity_v = [21.21e3, 21.21e3, 0];
%%
system.p1.force_v = [0, 0, 0];
system.p2.force_v = [0, 0, 0];

time = 24*3600*365*10;
system.dt = 3600;
system.centerOfMass(1, :) = (system.p1.mass*system.p1.position_v + ...
    system.p2.mass*system.p2.position_v)/(system.p1.mass + system.p2.mass);
% figure(1)
for i = 0:system.dt:time
    system = evaluate(system);
%     plot(i, sqrt(sum((system.p1.position_v - system.p2.position_v).^2)), 'gx')
%     hold on
%     plot(i, sqrt(sum(system.p2.force_v.^2)), 'ro')
end
% hold off

x1 = system.p1.trajectory_v_a(:, 1);
y1 = system.p1.trajectory_v_a(:, 2);
z1 = system.p1.trajectory_v_a(:, 3);
x2 = system.p2.trajectory_v_a(:, 1);
y2 = system.p2.trajectory_v_a(:, 2);
z2 = system.p2.trajectory_v_a(:, 3);

cmx = system.centerOfMass(:, 1);
cmy = system.centerOfMass(:, 2);
cmz = system.centerOfMass(:, 3);

figure(2)
axis auto
hold on
plot3(system.p1.trajectory_v_a(end, 1), system.p1.trajectory_v_a(end, 2), ...
        system.p1.trajectory_v_a(end, 3), 'gx', 'LineWidth', 2);
plot3(system.p2.trajectory_v_a(end, 1), system.p2.trajectory_v_a(end, 2), ...
        system.p2.trajectory_v_a(end, 3), 'rx', 'LineWidth', 2);
plot3(system.centerOfMass(end, 1), system.centerOfMass(end, 2), ...
        system.centerOfMass(end, 3), 'bx', 'LineWidth', 2);
plot3(x1, y1, z1, 'g', 'LineWidth', 2);
plot3(x2, y2, z2, 'r', 'LineWidth', 2);
plot3(cmx, cmy, cmz, 'b', 'LineWidth', 2);
grid on
hold off




% figure(3)
% axis auto
% hold on
% grid off
% plot3(system.p1.trajectory_v_a(end, 1), system.p1.trajectory_v_a(end, 2), ...
%         system.p1.trajectory_v_a(end, 3), 'gx', 'LineWidth', 2);
% plot3(system.p2.trajectory_v_a(end, 1), system.p2.trajectory_v_a(end, 2), ...
%         system.p2.trajectory_v_a(end, 3), 'rx', 'LineWidth', 2);
%   
% filename = 'testnew51.gif';
% for i = 1:1:length(system.p1.trajectory_v_a)
%     x1 = system.p1.trajectory_v_a(i, 1);
%     y1 = system.p1.trajectory_v_a(i, 2);
%     z1 = system.p1.trajectory_v_a(i, 3);
%     x2 = system.p2.trajectory_v_a(i, 1);
%     y2 = system.p2.trajectory_v_a(i, 2);
%     z2 = system.p2.trajectory_v_a(i, 3);
%     
%     plot3(x1, y1, z1, 'gx', 'LineWidth', 2);
%     plot3(x2, y2, z2, 'rx', 'LineWidth', 2);
%     
% %     M(i) = getframe;
% %     frame = getframe;
% %     im = frame2im(frame);
% %     [imind,cm] = rgb2ind(im,256);
% %     if i == 1;
% %         imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
% %     else
% %         imwrite(imind,cm,filename,'gif','WriteMode','append');
% %     end
%     mov(i) = getframe;
% end
% hold off
% movie2gif(mov, 'helix.gif', 'LoopCount', 0, 'DelayTime', 0) 
