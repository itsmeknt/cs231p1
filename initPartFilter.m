function [ model ] = initPartFilter( model, numParts )
cumIdx = 1;
for componentIdx=1:model.numcomponents
    energyMatrix = flattenByNorming(model.rootfilters{model.components{componentIdx}.rootindex}.w);
    

    for partIdx=1:numParts
        bbox = selectTopEnergyRegion(energyMatrix,width,heigth);
        
        energyMatrix = zeroOutEnergyRegion(energyMatrix, bbox);
        model = updateModelParameters(model, bbox, partIdx, componentIdx, cumIdx);
        cumIdx = cumIdx+1;
        
        symBbox = getSymmetricBbox(energyMatrix, bbox);
        if ~isempty(symBbox) && partIdx < numParts && computeOverlap(bbox, symBbox) < 0.5
            partIdx = partIdx+1;                                                            % we're adding a new, symmetric part
            
            energyMatrix = zeroOutEnergyRegion(energyMatrix, symBbox);
            model = updateModelParameters(model, width, height, symBbox, partIdx, componentIdx, cumIdx);
            cumIdx = cumIdx+1;
        end
    end
end
end


function [model] = updateModelParameters(model, partBbox, partIdx, componentIdx)
pIdx = model.components{componentIdx}.parts{partIdx}.partidx;
model.partfilters{pIdx}.w = model.rootfilters{model.components{componentIdx}.rootindex}.w(partBbox(1):partBbox(3), partBbox(2):partBbox(4), :);

dIdx = model.components{componentIdx}.parts{partIdx}.defidx;
model.defs{dIdx}.anchor = [bbox(1), bbox(2)];
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

function energyMatrix = zeroOutEnergyRegion(energyMatrix, bbox)
energyMatrix(bbox(1):bbox(3), bbox(2):bbox(4)) = 0;

function symBbox = getSymmetricBbox(energyMatrix, bbox)
n = size(energyMatrix,2);

leftSpace = bbox(1) - 0;
rightSpace = n - bbox(2);
% if bbox is already in the center
if (leftSpace == rightSpace)
    symBbox = [];                   % return empty
else
    symBbox = [rightSpace, bbox(2), n-leftSpace, bbox(4)];
end

function [matrix] = flattenByNorming(tensor)
sumsq = sum(tensor.*tensor), 3);
matrix = sqrt(sumsq);
end
