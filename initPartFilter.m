function [ model ] = initPartFilter( model, numParts )
cumIdx = 1;
for componentIdx=1:model.numcomponents
    energyMatrix = imresize(flattenByNorming(model.rootfilters{model.components{componentIdx}.rootindex}.w), 2, 'bicubic');
    
    idxOffset = 0;
    for partIdx=1:numParts
        if partIdx+idxOffset > numParts
            break;
        end
        pSize = size(model.partfilters{model.components{componentIdx}.parts{partIdx+idxOffset}.partidx}.w);
        bbox = selectTopEnergyRegion(energyMatrix,pSize(2),pSize(1));
        symBbox = getSymmetricBbox(energyMatrix, bbox);
        
        if ~isempty(symBbox) && partIdx+idxOffset+1 > numParts
            break;
        end
            
        
        energyMatrix = zeroOutEnergyRegion(energyMatrix, bbox);
        model = updateModelParameters(model, bbox, partIdx+idxOffset, componentIdx);
        cumIdx = cumIdx+1;
        
        if ~isempty(symBbox)
            idxOffset = idxOffset+1;                                                            % we're adding a new, symmetric part
            
            energyMatrix = zeroOutEnergyRegion(energyMatrix, symBbox);
            model = updateModelParameters(model, symBbox, partIdx+idxOffset, componentIdx);
            cumIdx = cumIdx+1;
        end
    end
end
end

function [model] = updateModelParameters(model, partBbox, partIdx, componentIdx)
pIdx = model.components{componentIdx}.parts{partIdx}.partidx;
energyMatrix = imresize(flattenByNorming(model.rootfilters{model.components{componentIdx}.rootindex}.w), 2, 'bicubic');
model.partfilters{pIdx}.w = energyMatrix(partBbox(1):partBbox(3), partBbox(2):partBbox(4), :);

dIdx = model.components{componentIdx}.parts{partIdx}.defidx;
model.defs{dIdx}.anchor = [partBbox(1), partBbox(2)];
end

function bbox = selectTopEnergyRegion(energyMatrix, width, height)
filter = ones(height, width);
conv = conv2(energyMatrix, filter);
summedEnergyMatrix = conv(height-1:size(conv,1)-height+1,width-1:size(conv,2)-width+1);
[maxCol, ys] = max(summedEnergyMatrix);
[maxVal, xs] = max(maxCol);
x = xs;
y = ys(x);
bbox = [x, y, x+width-1, y+height-1];
end

function energyMatrix = zeroOutEnergyRegion(energyMatrix, bbox)
energyMatrix(bbox(1):bbox(3), bbox(2):bbox(4)) = 0;
end

function symBbox = getSymmetricBbox(energyMatrix, bbox)
n = size(energyMatrix,2);

leftSpace = bbox(1) - 0;
rightSpace = n - bbox(3);
% if bbox is already in the center
if (leftSpace == rightSpace)
    symBbox = [];                   % return empty
else
    symBbox = [rightSpace, bbox(2), n-leftSpace, bbox(4)];
end
end

function [matrix] = flattenByNorming(tensor)
sumsq = sum((tensor.*tensor), 3);
matrix = sqrt(sumsq);
end
