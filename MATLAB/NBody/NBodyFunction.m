% This MATLAB code is based on a GNU Octave code created by:
% Harrison Brown
% University of Colorado Boulder, Boulder, Colorado, 80309, USA
% http://ccar.colorado.edu/asen5050/projects/projects_2013/Brown_Harrison/Code/Brown_H.pdf

function dx = NBodyFunction(t, x, m, G)
% t = integration time
% x = [ ... xi, yi, zi, vxi, vyi, vzi ...]
% m = [ m1, m2 ... mn]

G = 1.18555535802194e-04;
m = [1;332946;0.012303192];

n       = length(m);
% Each mass have six initial conditions: (x,y,z) (vx,vy,vz)
dx      = zeros(6*n, 1);
half    = 3*n;

r = zeros(n,n);

for i = 1:n
    for j = 1:n
        if((i == j) || (r(i,j) ~= 0))
            continue;
        else
            % r(i,j) = sqrt((xj - xi)^2 + (yj - yi)^2 + (zj - zi)^2)^3
            r(i,j) = sqrt((x(3*(j-1) + 1) - x(3*(i-1) + 1))^2 + ...
                     (x(3*(j-1) + 2) - x(3*(i-1) + 2))^2 + ...
                     (x(3*j) - x(3*i))^2)^3;
            r(j,i) = r(i,j);
        end
    end
end

% The change in position is equal to the current velocity, which is in the
% initial condition x vector
% x' = vx
% y' = vy
% z' = vz
% Filling in the velocities
for i = 1:half
    dx(i) = x(half + i);
end

% Building the accelerations
for i = 1:n
    %temp = 0;
    temp = [0, 0, 0];
    for j = 1:n
        if(i == j)
            continue;
        else
            % temp += mj * rij / |rij|^3
            rij_v = [   (x(3*(j-1) + 1) - x(3*(i-1) + 1)), ...
                        (x(3*(j-1) + 2) - x(3*(i-1) + 2)), ...
                        (x(3*j) - x(3*i))  ];
            temp = temp + (m(j)/r(i,j)) * rij_v;
            % temp = temp + (m(j)/r(i,j)) * (x(3*(j-1) + 1) - x(3*(i-1) + 1));
        end
    end
    temp = G * temp;
    % New velocity values
    dx(half+(3*(i-1))+1) = temp(1);
    dx(half+(3*(i-1))+2) = temp(2);
    dx(half+(3*i))       = temp(3);
end
end