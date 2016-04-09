function bodies = BH2DCalculateForces(tree, G)

    numBody = length(tree.rootNode.bodies);
    bodies = [];
    for i = 1:numBody
        currBody = tree.rootNode.bodies(i);
        currBody.acc = BH2DNodeBodyInteraction(tree.rootNode, currBody, G);
        if(isempty(bodies))
            bodies = currBody;
        else
            bodies = [bodies currBody];
        end
    end
end