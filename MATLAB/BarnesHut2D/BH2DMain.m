clear all
% close all
clc

% Settings

numBody = 2;
seed = 0;
startTime = 0;
endTime = 36500000;
stepTime = 10000;
GA = 8.890422785943706e-10;
% Create body vector with an Init function, that does randomization
% based on a seed
bodies = BH2DInit(numBody, seed);
% Build the tree
% Create a Tree, give the tree the body vector
% Recursively
% node = BH2DNode([0,0], width, bodies);
trajectories = zeros(numBody, 2,(endTime-startTime)/stepTime);
for j = 1:numBody
        trajectories(j, :, 1) = bodies(j).pos;
end
totalTime = 0;

for i = startTime:stepTime:endTime
    x = zeros(1, numBody);
    y = zeros(1, numBody);
    for j = 1:numBody
        x(j) = bodies(j).pos(:,1);
        y(j) = bodies(j).pos(:,2);
    end
    width = 2^ceil(log2(max([x, y])));
    
    tree = BH2DTree(width, bodies);

% Calculate forces, based on the tree
% Tree recalculate: get maximum and minimum of X and Y and max(X,Y) is the
% new width (round up to the next 2^ceil(log(maximum))
    tic
    bodies = BH2DCalculateForces(tree, GA);
    temp = toc;
    totalTime = totalTime + temp;
    for j = 1:numBody
        bodies(j).pos = bodies(j).pos + bodies(j).vel * stepTime;
        bodies(j).vel = bodies(j).vel + bodies(j).acc * stepTime;
        trajectories(j, :, i/stepTime + 1) = bodies(j).pos;
    end
end
% Statistics
disp('Total run time:')
disp(totalTime)
disp('Forces/second:')
disp(numBody*(numBody-1)*((endTime-startTime)/stepTime)/totalTime)
% Plotting
figure(1)
hold on;
cmap = hsv(numBody);
for i = 1:numBody
    x = [];
    y = [];
    for j = 1:(endTime-startTime)/stepTime
        x = [x trajectories(i,1,j)];
        y = [y trajectories(i,2,j)];
    end
    plot(x, y, 'x', 'Color', cmap(i,:), 'LineWidth', 2);
end
