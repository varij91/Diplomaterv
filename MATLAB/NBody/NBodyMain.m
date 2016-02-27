clear all
close all
clc

format long

numBody = 3;
dtime   = 100000;
eps     = 100000000;
seed    = 100;

system = NBodySystem(numBody, dtime, eps, seed);

numIteration = 50000;
tic
for i=1:numIteration
    system = system.evaluate();
end
totalTime = toc
evaluatePerSec = numIteration/totalTime
forceCalculationPerSec = evaluatePerSec*(numBody*numBody)

system.plot3();

