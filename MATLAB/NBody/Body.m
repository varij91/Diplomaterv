classdef Body
    properties
        mass
        position_v
        velocity_v
        acceleration_v
        trajectory_v_a
    end
    methods
        function obj = Body(mass, startPosition_v, startVelocity_v)
            if(isscalar(mass) && isnumeric(mass) && (mass > 0))
                obj.mass = mass;
            else
                error('mass : Mass must be a positive number.')
            end
            if(isnumeric(startPosition_v) && (length(startPosition_v) == 3))
                obj.position_v = startPosition_v;
            else
                error('startPosition_v : Starting position must be a 3 elemenet vector containing numbers.')
            end
            if(isnumeric(startVelocity_v) && (length(startVelocity_v) == 3))
                obj.velocity_v = startVelocity_v;
            else
                error('startVelocity_v : Starting velocity must be a 3 elemenet vector containing numbers.')
            end
            
            obj.trajectory_v_a(1, :) = startPosition_v;
            obj.acceleration_v = [0, 0, 0];
        end
        
        function obj = update(obj, dtime)
            % New position based on the old velocity values
            obj.position_v = obj.position_v + obj.velocity_v * dtime + 0.5 * obj.acceleration_v * dtime^2;
            
            % Adding new position to trajectory vector
            [m, n] = size(obj.trajectory_v_a);
            obj.trajectory_v_a(m + 1, :) = obj.position_v;
            
            % New velocity
            obj.velocity_v = obj.velocity_v + obj.acceleration_v * dtime;
        end
    end
end