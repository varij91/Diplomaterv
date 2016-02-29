clear all
close all
clc

format long

numBody = 4;
dtime   = 100000;
eps     = 100000000;
seed    = 100;

system = NBodySystem(numBody, dtime, eps, seed);
numIteration = 4;
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