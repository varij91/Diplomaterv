function [t, p] = NBodySingleStepBarnesHut(time, stepTime, mass, position, velocity, G, eps2)
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

end