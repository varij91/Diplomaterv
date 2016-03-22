function [runTime, t, x] = NBodySingleStep(algType, time, stepTime, bodies, G, eps)
    numBody = length(bodies);
    if(numBody < 2)
        error('SingleStep - At least 2 body needed to perform the simulation');
    end
    [mass, position_T, velocity_T] = NBodyToVectors(bodies);
    position = zeros(3,1,numBody);
    velocity = zeros(3,1,numBody);
    for i = 1:numBody
        position(:,:,i) = position_T(:,:,i)';
        velocity(:,:,i) = velocity_T(:,:,i)';
    end
    eps2 = eps^2;
    if(strcmp(algType, 'ALL_PAIRS'))
        tic
        [t, p] = NBodySingleStepAllPairs(time, stepTime, mass, position, velocity, G, eps2);
        runTime = toc;
    elseif(strcmp(algType, 'SELECTIVE'))
        tic
        [t, p] = NBodySingleStepSelectivePairs(time, stepTime, mass, position, velocity, G, eps2);
        runTime = toc;
    end
    
    cycles = length(t);
    x = zeros(1 + cycles, 3*numBody);
    for i = 1:numBody
        x(:, 3*(i-1) + 1) = p(1,:,i)';
        x(:, 3*(i-1) + 2) = p(2,:,i)';
        x(:, 3*i)         = p(3,:,i)';
    end
end