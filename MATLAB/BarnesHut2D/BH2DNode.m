classdef BH2DNode
%     enumeration
%         NorthEast (1)
%         SouthEast (2)
%         SouthWest (3)
%         NorthWest (4)
%     end
    properties (Constant)
        NorthEast = 1;
        SouthEast = 2;
        SouthWest = 3;
        NorthWest = 4;
    end
    properties
        center
        width
        
        virtualBody
        
        bodies
        sections
    end
    
    methods
        function obj = BH2DNode(center, width, bodies)
            obj.center  = center;
            obj.width   = width;
            obj.bodies  = bodies;
            if(~isempty(obj.bodies))
                obj = obj.distributeBody();
            end
        end
        function obj = addBody(obj, body)
            if(isempty(obj.bodies))
                obj.bodies = body;
            else
                obj.bodies = [obj.bodies, body];
            end
        end
        function section = getSection(obj, body)
            diffPos = body.pos - obj.center;
            if((diffPos(1) > 0) && (diffPos(2) >= 0))
                section = obj.NorthEast;
            elseif((diffPos(1) >= 0) && (diffPos(2) < 0))
                section = obj.SouthEast;
            elseif((diffPos(1) < 0) && (diffPos(2) >= 0))
                section = obj.SouthWest;
            else
                section = obj.NorthWest;
            end
        end
        function obj = distributeBody(obj)
            numBody = length(obj.bodies);
            if(numBody > 1)
                avgPos      = [0, 0];
                totalMass   = 0;
                for i = 1:numBody
                     totalMass = totalMass + obj.bodies(i).mass;
                     avgPos    = avgPos + obj.bodies(i).pos * obj.bodies(i).mass;
                end
                obj.virtualBody = BH2DBody(totalMass, avgPos/totalMass, [0, 0]);
                if(isempty(obj.sections))
                    cne = [obj.center(1) + obj.width/2, obj.center(2) + obj.width/2];
                    cse = [obj.center(1) + obj.width/2, obj.center(2) - obj.width/2];
                    csw = [obj.center(1) - obj.width/2, obj.center(2) - obj.width/2];
                    cnw = [obj.center(1) - obj.width/2, obj.center(2) + obj.width/2];
                    obj.sections = BH2DNode(cne, obj.width/2, []);
                    obj.sections(obj.SouthEast) = BH2DNode(cse, obj.width/2, []);
                    obj.sections(obj.SouthWest) = BH2DNode(csw, obj.width/2, []);
                    obj.sections(obj.NorthWest) = BH2DNode(cnw, obj.width/2, []);
                end
                for i = 1:numBody
                    section = obj.getSection(obj.bodies(i));
                    obj.sections(section) = obj.sections(section).addBody(obj.bodies(i));
                end
                
                for i = 1:4
                    obj.sections(i) = obj.sections(i).distributeBody();
                end
            end
        end
    end
end
