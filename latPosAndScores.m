function [Positions, MaxScores]=latPosAndScores(model,features,level)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the latent positions of the 
% parts given a feature pyramid and a model with a pretrained 
% root filter and part filters. It also computes the scores of the
% parts.

% The output Positions is a nx6 cell array of 2x1 vectors. n is the 
% number of model components.
% Positions{i,j}(1,1) and Positions{i,j}(2,1) are the x and y 
% coordinate of the latent part position of part j of component i.
% The output MaxScores is a nx6 cell array of scores.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_parts=6;
Positions=cell(model.numcomponents,6);
MaxScores=cell(model.numcomponents,6);

deform=zeros(size(features{level}(:,:,1)));   % deform stores deformation cost of  a block
a=1:size(deform,2);                           % a stores indices of columns in the feature window
b=1:size(deform,1);                           % b stores indices of rows in the feature window

% Iterate through each part to find the latent position
for i=1:model.numcomponents
    for j=1:num_parts
    
    partIdx=model.components{i}.parts{j}.partindex;
    defIdx=model.components{i}.parts{j}.defindex;
    partFilter=model.partfilters{partIdx};
    partAnchor=model.defs{defIdx};
    
    % Find cross correlation of the part filter with features
    C=imfilter(features{level},partFilter.w);
    
    % Find x and y distlacement matrix;
    
    deform=zeros(size(features{level}(:,:,1)));
    d_x=(a-partAnchor(2));                        % d_x stores the x deformation from the canonical position
    d_y=(b-partAnchor(1));                        % d_y stores the y defromation from the canonical position
    dx_sq=d_x.^2;
    dy_sq=d_y.^2;
    for m=1:size(deform,1)
        for n=1:size(deform,2)
            % Compute the deformation cost as the dot product of the
            % deformation feature and the deformations
            deform(m,n)=(partAnchor.w).*[d_x(n) d_y(m) dx_sq(n) dy_sq(m)];
        end
    end
    
    C=C-deform;
    [dummy_max,dummy_ind]=max(C);
    [MaxScores{i,j},Positions{i,j}(1,1)]=max(dummy_max);
    Positions{i,j}(2,1)=dummy_ind(dummy_max);
    
    end
end


