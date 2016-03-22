function [t, p] = NBodySingleStepSelectivePairs(time, stepTime, mass, position, velocity, G, eps2)
    distanceLimit2 = 1e8;
    numBody = length(mass);
    % Compute the forces and the next position for each body based
    % on the given timestep
    r = zeros(numBody,numBody);
    for i = 1:numBody
        for j = 1:numBody
            if((i == j) || (r(i,j) ~= 0))
                continue;
            else
                r_v    = position(:,:,j) - position(:,:,i);
                r(i,j) = sqrt(((sum(r_v.^2)+eps2)^3));
                r(j,i) = r(i,j);
            end
        end
    end
    t = time(1):stepTime:time(2);
    cycles = length(t);
    p = zeros(3, 1 + cycles, numBody);
    p(:,1,:) = position;
    
    for k = 1:cycles
        for i = 1:numBody
            p(:,k + 1,i) = p(:,k,i) + velocity(:,:,i) * stepTime;
            acc_v = [0; 0; 0];
            for j = 1:numBody
                if(i == j)
                    continue;
                end
                if(r(i,j) <= distanceLimit2)
                    acc_v = acc_v + G * mass(j) * (p(:, k, j) - p(:, k, i)) / r(i,j);
                else
                    continue;
                end
            end
            velocity(:,:,i) = velocity(:,:,i) + acc_v * stepTime;
        end
    end
end