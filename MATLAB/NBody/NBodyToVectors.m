function [mass, pos, vel] = NBodyToVectors(bodies)
% Utility function - Unwrap Body type objects into vectors.

    numBody = length(bodies);
    if(~isnumeric(numBody) || numBody < 1)
        error('Input bodies must be numeric and positive');
    end
    mass = zeros(1,numBody);
    pos = zeros(1, 3, numBody);
    vel = zeros(1, 3, numBody);
    for i = 1:numBody
        if(isa(bodies(i), 'NBodyParticle'))
            mass(i) = bodies(i).mass;
            pos(:,:,i) = bodies(i).position_v;
        	vel(:,:,i) = bodies(i).velocity_v;
        else
            error('Input bodies must be instances of Body class');
        end
    end
end