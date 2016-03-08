function [runTime, t, x] = NBodyOde(algType, time, stepTime, bodies, G, eps, opts)
    if(length(bodies) < 2)
        error('ODE - At least 2 body needed to perform the simulation');
    end
    [mass, position, velocity] = NBodyToVectors(bodies);
    x0 = zeros(1, 6 * length(bodies));
    half = 3 * length(bodies);
    for i = 1 : length(bodies)
        x0(3*(i-1) + 1)        = position(1, 1, i);
        x0(3*(i-1) + 2)        = position(1, 2, i);
        x0(3*(i))              = position(1, 3, i);
        x0(half + 3*(i-1) + 1) = velocity(1, 1, i);
        x0(half + 3*(i-1) + 2) = velocity(1, 2, i);
        x0(half + 3*(i))       = velocity(1, 3, i);
    end
    eps2 = eps^2;
    if(strcmp(algType, 'BARNES_HUT'))
    elseif(strcmp(algType, 'ALL_PAIRS'))
        tic
        wrapper = @(t,x, opts) NBodyOdeAllPairs(t, x, mass, G, eps2);
        [t, x]  = ode45(wrapper, time, x0, opts);
% x0 = [1;0;0;0;0;0;0.99743;0;0;0;0.9;0;0;0;0;0;0.5;0.2148058584];
% x0 = [1;0;0;0;0;0;0.99743;0;0;0;2;0;0;0;0;0;2;0.8];
%     x0 = [0 0 0 1.000000000000000 0 0 0.997430000000000 0 0 0 0 0 0 0.017210550000000 0 0 0.017210558354552 0.000588106388501];
%     [t,x] = ode45(@NBodyOdeAllPairs, time, x0);
% ode45(@NBodyOdeAllPairs, time, x0);
        runTime = toc;
    else
        error('Invalid algorithm type.')
    end
end