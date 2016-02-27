classdef Particle
    properties
        position_v      % (x, y, z) in [m]
        trajectory_v_a  % (x, y, z) in [m]
        mass            % [kg]
        velocity_v      % [m/s]
        force_v         % [kg*m/s^2] = [N]
    end
    methods
        function obj = setPosition(obj, newPosition_v)
            [m, n] = size(newPosition_v);
            if(n == 3)
                obj.position_v = newPosition_v;
            else
                error('newPosition must be row vector and contain 3 (x, y, z) element.');
            end
        end
    end
end