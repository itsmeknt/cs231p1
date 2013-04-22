function [ model ] = initPartFilter( model, numParts )
cumIdx = 1;
for componentIdx=1:model.numcomponents
    energyMatrix = flattenByNorming(model.rootfilters{model.components{componentIdx}.rootindex}.w);
    
    rDim = model.rootfilters{model.components{componentIdx}.rootindex}.size;
    rArea = rDim(1)*rDim(2);
    regionArea = 0.8*rArea;
    
    width = sqrt(regionArea*(rDim(2)/rDim(1));
    height = width*(rDim(1)/rDim(2));
    for partIdx=1:numParts
        bbox = selectTopEnergyRegion(energyMatrix,
        
        energyMatrix = zeroOutEnergyRegion(energyMatrix, bbox);
        updateModelParameters(model, width, height, bbox, partIdx, componentIdx, cumIdx);
        cumIdx = cumIdx+1;
        
        symBbox = getSymmetricBbox(energyMatrix, bbox);
        if ~isempty(symBbox) && partIdx < numParts && computeOverlap(bbox, symBbox) < 0.5
            partIdx = partIdx+1;                                                            % we're adding a new, symmetric part
            
            energyMatrix = zeroOutEnergyRegion(energyMatrix, symBbox);
            updateModelParameters(model, width, height, symBbox, partIdx, componentIdx, cumIdx);
            cumIdx = cumIdx+1;
        end
    end
end
end


function [model] = updateModelParameters(model, width, height, partBbox, partIdx, componentIdx, cumIdx)
model.partfilters{cumIdx}.w = zeros(height, width, 31);
model.partfilters{cumIdx}.blocklabel = blockIdx;
blockIdx = 2*cumIdx+1;

model.defs{cumIdx}.anchor = [bbox(1), bbox(2)];
model.defs{cumIdx}.w = zeros(4, 1);
model.defs{cumIdx}.blocklabel = blockIdx;
blockIdx = 2*cumIdx+2;

model.components{componentIdx}.parts{partIdx}.partindex = cumIdx;
model.components{componentIdx}.parts{partIdx}.defindex = cumIdx;
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
