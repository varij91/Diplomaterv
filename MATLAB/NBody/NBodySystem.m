classdef NBodySystem
   properties (GetAccess=private)
        bodies_a
        evaluateCount
   end
   properties
        % General properties
        numBody
        dtime
        eps2
        
        seed
    end
    properties (Constant)
        G = 6.67408e-11     % 6.67408 x 10-11 m3 kg-1 s-2
    end
    methods
        function obj = NBodySystem(numBody, dtime, eps, seed)
            if(isnumeric(numBody) && (numBody > 0))
                obj.numBody = floor(numBody);
            else
                error('numBody : Number of bodies must be a positive integer.')
            end
            if(isnumeric(dtime) && (dtime > 0))
                obj.dtime = dtime;
            else
                error('dtime : Time step must be a positive number.')
            end
            if(isnumeric(eps))
                obj.eps2 = eps^2;
            else
                error('eps : Softening factor must be a number.')
            end
            if(isnumeric(seed) && (seed >= 0))
                obj.seed = floor(seed);
            else
                error('seed : Intitial seed must be a non-negative integer.')
            end
            
            obj.evaluateCount = 0;

            obj = obj.init();
        end
        function obj = init(obj)
            % Initial random values for the bodies
            rng(obj.seed);
            for i = 1:obj.numBody
                % Mass - Uniform distribution
                % [1e25, 10e35]
                m = (1 + 9*rand(1)) * 10^(round(25 + 10*rand(1)));
                
                % Position - Normal distribution
                pos = 2e13 * randn(1,3);
                
                % Velocity - Normal distribution
                vel = 1e4 * randn(1,3);
                
                b = Body(m, pos, vel);
                if(isempty(obj.bodies_a))
                    obj.bodies_a = b;
                else
                    obj.bodies_a(i) = b;
                end
            end
%             b = Body(2e30, [0,0,0], [0,0,0]);
%             obj.bodies_a = b;
%             b = Body(6e24, [0, 1.5e11, 0], [21.21e3, 21.21e3, 0]);
%             obj.bodies_a(2) = b;
        end
        function obj = evaluate(obj)
            % Compute the forces and the next position for each body based
            % on the given timestep
            for i = 1:obj.numBody
                acc_v = [0, 0, 0];
                for j = 1:obj.numBody
                    rij_v = obj.bodies_a(j).position_v - obj.bodies_a(i).position_v;
                    rabs  = sqrt((sum(rij_v.^2) + obj.eps2)^3);
                    acc_v = acc_v + obj.G * obj.bodies_a(j).mass * rij_v / rabs;
                end
                obj.bodies_a(i).acceleration_v = acc_v;
            end
            for i = 1:obj.numBody
                obj.bodies_a(i) = obj.bodies_a(i).update(obj.dtime);
            end
            obj.evaluateCount = obj.evaluateCount + 1;
        end
        
        function obj = evaluate2(obj)
            % Compute the forces and the next position for each body based
            % on the given timestep
            % Precalculating distatance: instead of N^2 computation -->
            % N^2/2
            rij_m = zeros(obj.numBody, obj.numBody);
            sqrteps = sqrt(obj.eps2^3);
            for i = 1:obj.numBody
                for j = 1:obj.numBody
                    if(i == j)
                        rij_m(i,j) = sqrteps;
                    elseif(rij_m(i,j) ~= 0)
                        continue;
                    else
                        rij_v = obj.bodies_a(j).position_v - obj.bodies_a(i).position_v;
                        rij_m(i,j) = sqrt((sum(rij_v.^2) + obj.eps2)^3);
                        rij_m(j,i) = rij_m(i,j);
                    end 
                end
            end
            for i = 1:obj.numBody
                acc_v = [0, 0, 0];
                for j = 1:obj.numBody
                    rij_v = obj.bodies_a(j).position_v - obj.bodies_a(i).position_v;
                    acc_v = acc_v + obj.G * obj.bodies_a(j).mass * rij_v / rij_m(i,j);
                end
                obj.bodies_a(i).acceleration_v = acc_v;
            end
            for i = 1:obj.numBody
                obj.bodies_a(i) = obj.bodies_a(i).update(obj.dtime);
            end
            obj.evaluateCount = obj.evaluateCount + 1;
        end
        
        function plot3(obj)
            figure
            hold on
            cmap = hsv(obj.numBody);
            for currBody = 1:obj.numBody
                x = obj.bodies_a(currBody).trajectory_v_a(:, 1);
                y = obj.bodies_a(currBody).trajectory_v_a(:, 2);
                z = obj.bodies_a(currBody).trajectory_v_a(:, 3);
%                 plot3(x, y, z, 'x', 'Color', cmap(currBody,:), 'LineWidth', 1);
                plot3(x, y, z, 'Color', cmap(currBody,:), 'LineWidth', 2);
                
                x = obj.bodies_a(currBody).trajectory_v_a(end, 1);
                y = obj.bodies_a(currBody).trajectory_v_a(end, 2);
                z = obj.bodies_a(currBody).trajectory_v_a(end, 3);
                plot3(x, y, z, 'x', 'Color', cmap(currBody,:), 'LineWidth', 2);
            end
            hold off
            grid on
        end
    end
end