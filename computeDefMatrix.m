function deform=computeDefMatrix(rootSize,partSize,partDef)

% This function computes the deformation cost matrix given the root filter
% size and a particular part deformation model (deformation weights and 
% anchor position).

deform=zeros(2*rootSize(1)-partSize(1)+1, 2*rootSize(2)-partSize(2)+1);

% Get the deformation weights and the anchor position of the part

v=partDef.anchor; % Multiply canonical coordinates by 2 for double resolution for part filters
w=partDef.w;


% Initalize displacements
y_range=1:2*rootSize(1)-partSize(1)+1;
x_range=1:2*rootSize(2)-partSize(2)+1;
dy=y_range-v(1)*ones(1,2*rootSize(1)-partSize(1)+1);
dx=x_range-v(2)*ones(1,2*rootSize(2)-partSize(2)+1);
dy_sq=dy.^2;
dx_sq=dx.^2;



for i=1:2*rootSize(1)-partSize(1)+1
    for j=1:2*rootSize(2)-partSize(2)+1
        d=[dy(1,i); dx(1,j); dy_sq(1,i); dx_sq(1,j)];
        deform(i,j)=sum(w*d);
    end
end