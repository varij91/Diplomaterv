function bodies = NBodyInit(initType, numBody, seed)
    if(~isnumeric(numBody))
        error('Number of body must be numeric')
    end
    numBody = floor(numBody);
    if(numBody < 1)
        error('Number of body must be positive')
    end
    
    if(~isnumeric(seed) || seed < 0)
        error('Seed must be numeric and positive')
    end
    seed = floor(seed);
    
    switch initType
        % Two bodies
        case '2_STEADY'
        case '2_HYPERBOLIC'
        case '2_ELLIPTIC'
        case '2_CIRCLE'
        case '2_CIRCLE_2' %mindkettõ mozog, eq mess
        case '2_HELIX'   % elõre megy a csillag
        case '2_HELIX_EQUAL_MASS'  % dugóhúzó felfele
        case '2_RANDOM'
        % Three bodies
        case '3_STEADY'
        case '3_SEL'
            sun     = NBodyParticle(332946,      [0, 0, 0],       [0, 0, 0]);
            earth   = NBodyParticle(1,           [1, 0, 0],       [0, 0.01721055, 0]);
            luna    = NBodyParticle(0.012303192, [0.99743, 0, 0], [0, 0.017210558354552, 5.881063885010267e-04]);
            bodies = [sun, earth, luna];
        case '3_EQUAL_MASS'
        case '3_RANDOM'
        % N bodies
        case 'N_RANDOM'
            rng(seed);
            % Mass - Uniform distribution
            m = 10*rand(1, numBody) .* (10.^round(7*rand(1, numBody)));
            % Position - Normal distribution
            pos = 10.*randn(1, 3, numBody) .* (10.^round(2.*rand(1,3,numBody)));
            % Velocity - Normal distribution
            vel = 0.01.*randn(1, 3, numBody);
%             vel = zeros(1,3,numBody);
            for i = 1:numBody
                if(i == 1)
                    bodies = NBodyParticle(m(i), pos(:,:,i), vel(:,:,i));
                else
                    bodies = [bodies NBodyParticle(m(i), pos(:,:,i), vel(:,:,i))];
                end
            end
        otherwise
            error('Invalid initialization type.');
    end
end
% fun values
% x0 = [1;0;0;0;0;0;0.99743;0;0;0;0.9;0;0;0;0;0;0.5;0.2148058584];