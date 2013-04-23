function [ croppedFeatPyramid ] = cropFeaturePyramid(featPyramid, scales)
% inverse of getBoundingBox
x1 = bbox(1);
y1 = bbox(2);
x2 = bbox(3);
y2 = bbox(4);

scale = (x2-x1+1)/rsize(2);
x = padx + (x1-1)/scale;
y = pady + (y1-1)/scale;
end

