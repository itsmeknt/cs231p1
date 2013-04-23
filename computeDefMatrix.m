function deform=computeDefMatrix(rootSize,partDef)

% This function computes the deformation cost matrix given the root filter
% size and a particular part deformation model (deformation weights and 
% anchor position).

deform=zeros(2*rootSize);

% Get the deformation weights and the anchor position of the part

v=2*partDef.anchor; % Multiply canonical coordinates by 2 for double resolution for part filters
w=partDef.w;

% Initalize displacements
dy=abs(1:2*rootSize(1)-v(1));
dx=abs(1:2*rootSize(2)-v(2));
dy_sq=dy.^2;
dx_sq=dx.^2;

d=[dy dx dy_sq dx_sq];

for i=1:2*rootSize(1)
    for j=1:2*rootSize(2)
        deform(i,j)=w.*d;
    end
end