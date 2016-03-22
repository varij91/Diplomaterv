% Used unit system: Astronomical system of units (IAU)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Astronomical system of units
% The system was developed because of the difficulties in measuring and
% expressing astronomical data in International System of Units (SI units).
% In particular, there is a huge quantity of very precise data relating to
% the positions of objects within the solar system which cannot
% conveniently be expressed or processed in SI units.
% 
% The astronomical system of units is a tridimensional system, in that it
% defines units of length, mass and time.
% 
% The astronomical unit of time is the day [D], defined as 86400 seconds.
% 365.25 days make up one Julian year.
% 
% Earth mass [EM] is the unit of mass equal to that of the Earth.
% 1 [EM] = 5.9742e24 kg.
% 
% The astronomical unit [AU] of length is now defined as exactly
% 149 597 870 700 meters.
% 
% With this units the gravitational constant's (G = 6.67408e-11
% m^3*kg^-1*s^-2) will become GA = 8.890422785943706e-10 AU^3*EM^-1*D^-2
% Chaging units: GA = G * ME * D^2 / AU^3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc

format long
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETTINGS
% Algorithm type
%   'All_PAIRS'
%   'SELECTIVE' (ONLY IN SINGLESTEP)
%   'BARNES_HUT'
algType     = 'ALL_PAIRS';
% Simulation solver
%   'SINGLESTEP'
%   'ODE'
% simSolver   = 'SINGLESTEP';
simSolver   = 'SINGLESTEP';
opts = odeset('reltol', 1e-3, 'abstol', 1e-6, 'refine', 1);
% Simulation's goal
%   'EVALUATE'
%   'RUNTIME'
simGoal     = 'EVALUATE';
simCycle    = 50;
% Simulation type
%   '3_SEL'     = Sun-Earth-Luna simulation
%   'N_RANDOM'  = numBody sized universe with random initial values
simType     = 'N_RANDOM';
% Number of bodies (in some simulation cases has no effect)
numBody     = 4;
% Simulation time - [D]
startTime   = 0;
endTime     = 350000;
% Simulation advance in time - [D]
% (Not in ODE)
stepTime    = 3;
% Softening factor - [AU]
eps         = 0.00001;
% Set initialization seed to an non-negative integer
seed        = 0;
% Write results into file
logging     = 'OFF';
logFile     = 'NBodyLogs.txt';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIALIZATION
% Simulation time - [D]
time = [startTime, endTime];
% Number of simulations
switch simGoal
    case 'EVALUATE'
        numSim = 1;
    case 'RUNTIME'
        numSim = simCycle;
    otherwise
        numSim = 1;
end
% Gravitational constant - [(AU^3)*(EM^-1)*(D^-2)]
GA = 8.890422785943706e-10;
% Bodies
if(regexp(simType, '2_*'))
    numBody = 2;
elseif(regexp(simType, '3_*'))
    numBody = 3;
end
bodies = NBodyInit(simType, numBody, seed);
% Runtime - [s]
totalTime = zeros(1,numSim);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SIMULATIONS
for i = 1:numSim
    switch simSolver
        case 'SINGLESTEP'
            [totalTime(i), t, x] = NBodySingleStep(algType, time, stepTime, bodies, GA, eps);
        case 'ODE'
            [totalTime(i), t, x] = NBodyOde(algType, time, stepTime, bodies, GA, eps, opts);
        otherwise
    end
end
totalTime = sum(totalTime);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOGGING
if(strcmp(logging,'ON'))
    NBodyLogging(logFile, simType, simSolver, algType, numBody, startTime, ...
    endTime, stepTime, seed, eps, totalTime, totalTime/numSim);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUALIZATION
% Runtime
disp('Total run time:')
disp(totalTime)
disp('Average run time:')
disp(totalTime/numSim)
disp('Forces/second:')
[m,n] = size(x);
avgsteptime = (endTime-startTime)/m;
disp(numBody*(numBody-1)*((endTime-startTime)/avgsteptime)/totalTime)
% Plots
figure
hold on;
cmap = hsv(numBody);
for i = 1:numBody
    plot3(x(:,3*(i-1)+1), x(:,3*(i-1)+2), x(:, 3*i), 'Color', cmap(i,:), 'LineWidth', 2);
%     plot3(x(:,3*(i-1)+1), x(:,3*(i-1)+2), x(:, 3*i), 'o', 'Color', cmap(i,:), 'LineWidth', 2);
    plot3(x(end,3*(i-1)+1), x(end,3*(i-1)+2), x(end, 3*i), 'x', 'Color', cmap(i,:), 'LineWidth', 2);
end
hold off;