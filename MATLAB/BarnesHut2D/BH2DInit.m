function bodies = BH2DInit(numBody, seed)
    if(numBody < 2)
        error('NumBody must be at least 2');
    end
    rng(seed);
    % Mass - Uniform distribution
    m = 10*rand(1, numBody) .* (10.^round(7*rand(1, numBody)));
    % Position - Normal distribution
    pos = 10.*randn(1, 2, numBody) .* (10.^round(2.*rand(1,2,numBody)));
    % Velocity - Normal distribution
    vel = 0.01.*randn(1, 2, numBody);
%           vel = zeros(1,2,numBody);
    for i = 1:numBody
        if(i == 1)
            bodies = BH2DBody(m(i), pos(:,:,i), vel(:,:,i));
        else
            bodies = [bodies BH2DBody(m(i), pos(:,:,i), vel(:,:,i))];
        end
    end
end