function [bbox] = getBoundingBox(x, y, scale, padx, pady, rsize)

% Gets original pixel coordinate bounding box of a block in the
% pyramid level.
%
% x = the x index of the block at a pyramid level
% y = the y index of the block at a pyramid level
% scale = the resolution scale of the pyramid level
% padx = size of padding along x axis
% pady = size of padding along y axis
% rsize = root filter size of the component of the model used to do
% the convolution on the pyramid level
%
% returns the original pixel coordinate bounding box

x1 = (x-padx)*scale+1;
y1 = (y-pady)*scale+1;
x2 = x1 + rsize(2)*scale - 1;
y2 = y1 + rsize(1)*scale - 1;
bbox = [x1, y1, x2, y2];