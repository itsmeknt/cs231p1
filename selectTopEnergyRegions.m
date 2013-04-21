function [bboxes] = selectTopEnergyRegions(energyMatrix, width, height, numRegions)
bboxes = zeros(numRegions, 4);
energyMatrixCopy = energyMatrix;
for i=1:numRegions
    bbox = selectTopEnergyRegion(energyMatrixCopy, width, height);
    bboxes(i) = box;
    energyMatrixCopy(bbox(1):bbox(3), bbox(2):bbox(4)) = 0;
end
end






function bbox = selectTopEnergyRegion(energyMatrix, width, height)
filter = ones(height, width);
conv = conv2(energyMatrix, filter);
summedEnergyMatrix = conv(height-1:size(conv,1)-heigth+1,width-1:size(conv,2)-width+1);
[maxCol, ys] = max(summedEnergyMatrix);
[maxVal, xs] = max(maxCol);
x = xs;
y = ys(x);
bbox = [x, y, x+width-1, y+height-1];
end