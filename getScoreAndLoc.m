function [partScore,partLocation]=getScoreAndLoc(convolved,x,y,deform,rootSize)

% This function computes the best location of a single part, and net score
% (Fi*Phi - deformation) given the location (x, y) of the root, the 
% convolution score of the part filter with the feature pyramid
% (convolved), the deformation cost matrix and the size of the root filter.

% crop the convolved score matrix to find the region covered by the root filter

clipped=convolved((2*y):(2*y+2*rootSize(1)-1),(2*x):(2*x+2*rootSize(2)-1));
Net_Score=clipped-deform;
[max_x,ind_x]=max(Net_Score);
[partScore,partLocation(1)]=max(max_x);
partLocation(2)=ind_x(partLocation(1));


