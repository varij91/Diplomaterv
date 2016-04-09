function acc = BH2DNodeBodyInteraction(node, body, G)
    eps2 = 1e-12;
    if(length(node.bodies) == 1)
        r = node.bodies.pos - body.pos;
        ra3 = sqrt(sum(r.^2) + eps2)^3;
        acc = G * node.bodies.mass * r / ra3;
    elseif(length(node.bodies) > 1)
        acc = [0, 0];
        for i = 1:4
            acc = acc + BH2DNodeBodyInteraction(node.sections(i), body, G);
        end
    else
        acc = [0, 0];
    end
end