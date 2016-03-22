clear all
% close all
clc

% Settings

numBody = 5;
seed = 0;
startTime = 0;
endTime = 365;
stepTime = 1;
GA = 8.890422785943706e-10;
% Create body vector with an Init function, that does randomization
% based on a seed
bodies = BH2DInit(numBody, seed);
% Build the tree
% Create a Tree, give the tree the body vector
% Recursively
% node = BH2DNode([0,0], width, bodies);
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
%     bodies = BH2DCalculateForces(tree, GA);
end

% Statistics

% Plotting
