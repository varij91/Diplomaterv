classdef BH2DBody
    properties
        mass
        vel
        pos
        acc
    end
    methods       
        function obj = BH2DBody(mass, pos, vel)
            obj.mass = mass;
            obj.pos  = pos;
            obj.vel  = vel;
            obj.acc  = [0, 0];
        end
    end
end