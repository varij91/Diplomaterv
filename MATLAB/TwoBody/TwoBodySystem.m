classdef TwoBodySystem
    properties
        p1 = Particle
        p2 = Particle
        
        dt              % timestep
        centerOfMass
    end
    properties (Constant)
        G = 6.67408e-11 %6.67408 x 10-11 m3 kg-1 s-2
    end
    methods
        function obj = evaluate(obj)
            % Init values
            m1 = obj.p1.mass;
            m2 = obj.p2.mass;
            
            r_v = obj.p2.position_v - obj.p1.position_v;
            r = sqrt(sum(r_v.^2));
            
            % Force
            F  = obj.G * m1 * m2 * r_v / r^3;
            F1 = F;
            F2 = -F;
            
            % Acceleration
            a1 = F1/m1;
            a2 = F2/m2;
            
            % Position
            r1 = obj.p1.position_v + obj.p1.velocity_v * obj.dt + 0.5 * a1 * obj.dt^2;
            r2 = obj.p2.position_v + obj.p2.velocity_v * obj.dt + 0.5 * a2 * obj.dt^2;
            
            % Velocity
            v1 = obj.p1.velocity_v + a1 * obj.dt;
            v2 = obj.p2.velocity_v + a2 * obj.dt;

            % Write results
            [t11, t12] = size(obj.p1.trajectory_v_a);
            [t21, t22] = size(obj.p2.trajectory_v_a);
            obj.p1.position_v = r1;
            obj.p2.position_v = r2;
            obj.p1.trajectory_v_a(t11+1, :) = r1;
            obj.p2.trajectory_v_a(t21+1, :) = r2;
            obj.centerOfMass(t11+1, :) = (m1*r1 + m2*r2)/(m1 + m2);
            obj.p1.velocity_v = v1;
            obj.p2.velocity_v = v2;
            obj.p1.force_v = F1;
            obj.p2.force_v = F2;
        end
    end
end