function [detectionsAboveThreshold, detectionsAtThreshold, detectionsBelowThreshold] = detect(featPyramid, scales, model, threshold)

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

detectionsAbove = [];
detectionsAt = [];
detectionsBelow = [];
for pLevelIdx = 1:length(scales)
    feat = featPyramid{pLevelIdx};
    scale = scales(pLevelIdx);
    for cIdx = 1:model.numcomponents
        rootSize = model.rootfilters{model.components{cIdx}.rootindex}.size;
        
        % compute score
        score = feat;
        padx;
        pady;
        
        % threshold scores
        I = find(score > thresh);
        detectionsAbove = [detectionsAbove; formatDetections(score, I, cIdx, scale, padx, pady, rootSize)];
        I = find(score == thresh);
        detectionsAt = [detectionsAt; formatDetections(score, I, cIdx, scale, padx, pady, rootSize)];
        I = find(score < thresh);
        detectionsBelow = [detectionsBelow; formatDetections(score, I, cIdx, scale, padx, pady, rootSize)];
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

