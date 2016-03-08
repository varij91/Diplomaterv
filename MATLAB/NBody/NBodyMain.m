clear all
close all
clc

format long

numBody = 3;
dtime   = 100000;
eps     = 100000000;
seed    = 100;

system = NBodySystem(numBody, dtime, eps, seed);
numIteration = 34000;
tic
for i=1:numIteration
    system = system.evaluate();
end
totalTime = toc
evaluatePerSec = numIteration/totalTime
forceCalculationPerSec = evaluatePerSec*(numBody*numBody)
system.plot3();

system2 = NBodySystem(numBody, dtime, eps, seed);
tic
for i=1:numIteration
    system2 = system2.evaluate2();
end
totalTime = toc
evaluatePerSec = numIteration/totalTime
forceCalculationPerSec = evaluatePerSec*(numBody*numBody)
system2.plot3();
%%
% TODO: wrapper for the ODE versions
% Must separate initialization and running simulation
% main --> init bodies with options(random, SEL, custom(input))--> gives 
% them to the system and to the ode wrapper

% This is a script that simulates the Sun, Earth, Luna system over the
% period of one tropical year
G = 1.18555535802194e-04; % Gravitational Constant in astronomical units
time = [0 1]; % From 0 to 1 year
x0 = [1;0;0;0;0;0;0.99743;0;0;0;6.286156439;0;0;0;0;0;6.286156439;0.2148058584];
% Positions of the Earth-Sun-Moon, then the Velocities of the Earth-Sun-Moon
% AU and (AU/yr)
% x0 = [1;0;0;0;0;0;0.99743;0;0;0;0.017;0;0;0;0;0;0.017;0.0005];
m = [1;332946;0.012303192]; % The masses of the Earth, Sun, and Moon (EM)
% tol = 10e-7;
% opts = odeset('reltol',tol,'abstol',tol,'refine',1);
[t,x] = ode45(@NBodyFunction, time, x0, opts);

% Plotting Functions
figure
hold on;
plot3(x(:,1),x(:,2),x(:,3),'b-'); % Earth
plot3(x(:,4),x(:,5),x(:,6),'r*');  % Sun
plot3(x(:,7),x(:,8),x(:,9),'g-'); % Moon