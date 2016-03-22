classdef BH2DTree
    properties
        rootNode
    end
    methods
        function obj = BH2DTree(width, bodies)
            obj.rootNode = BH2DNode([0, 0], width, bodies);
        end
    end
end