function [detectionsAboveThreshold, detectionsAtThreshold, detectionsBelowThreshold bestRootLoc bestPartLoc bestRootLevel bestComponentIdx bestScore] = detect(featPyramid, scales, model, threshold)

% detects objects in feat pyramid
%
% inputs:
% featPyramid - feature pyramid from featpyramid.m
% scales - scales of the feature pyramid from featpyramid.m
% model - model to use for detection
% threshold - detection threshold
% getScoresAbove - boolean value. If true, grabs all bounding box strictly
% above threshold. If false, grabs all bounding box strictly below
% threshold.
%
% output:
% detections = an array of detection structs. detection structs have the
% following member variables:
%   .rootBbox - bounding box ([x1, y1, x2, y2]) of root position
%   .partBbox - a matrix of bounding box where each row represents the bounding
%               box of a part. So it is a px4 matrix. The index for the
%               part in this matrix is thes ame as the index in
%               model.filters.
%   .score - score of the detection
%   .component - the component used for the detection

detectionsAboveThreshold = [];
detectionsAtThreshold = [];
detectionsBelowThreshold = [];
% compute score
[bestComponentIdx,bestRootLoc,bestPartLoc,bestRootLevel,bestScore,Array]=latent(model,featPyramid,scales);

for pLevelIdx = 1:length(scales)
    scale = scales(pLevelIdx);
    for cIdx = 1:model.numcomponents
        rootSize = model.rootfilters{model.components{cIdx}.rootindex}.size;
        
        score = Array{cIdx, pLevelIdx};
       
        % threshold scores
        Iabove = find(score > threshold);
        detectionsAboveThreshold = [detectionsAboveThreshold; formatDetections(score, Iabove, cIdx, scale, model.padx, model.pady, rootSize)];
        Iat = find(score == threshold);
        detectionsAtThreshold = [detectionsAtThreshold; formatDetections(score, Iat, cIdx, scale, model.padx, model.pady, rootSize)];
        Ibelow = ~(Iabove | Iat);
        detectionsBelowThreshold = [detectionsBelowThreshold; formatDetections(score, Ibelow, cIdx, scale, model.padx, model.pady, rootSize)];
    end
end
end


function detections = formatDetections(score, I, componentIdx, scale, padx, pady, rootSize)
detections = [];
[Y, X] = ind2sub(size(score), I);
for i = 1:length(I)
    x = X(i);
    y = Y(i);
    rootBbox = getBoundingBox(x, y, scale, padx, pady, rootSize);
    entry.rootBbox = rootBbox;
    entry.partBbox = [];
    entry.score = score(I(i));
    entry.component = componentIdx;
    detections = [detections; entry];
end
end

