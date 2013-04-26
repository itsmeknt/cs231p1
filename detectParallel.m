function [detectionsAboveThreshold, detectionsAtThreshold bestRootLoc bestPartLoc bestRootLevel bestComponentIdx bestScore] = detectParallel(posOrNegs, model, threshold)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[dummy, scales] = loadFeaturePyramidCache(posOrNegs(1).id);
n = length(posOrNegs);
feats = cell(1,n);
idxs = 1:n;
%matlabpool open;
%parfor p=1:n
%    feats{p} = loadFeaturePyramidCache(posOrNegs(p).id);
%end

detectionsAboveThreshold = cell(1,n);
detectionsAtThreshold = cell(1,n);
bestRootLoc = cell(1,n);
bestPartLoc = cell(1,n);
bestRootLevel = cell(1,n);
bestComponentIdx = cell(1,n);
bestScore = cell(1,n);

dtic = tic;
for i=1:n
%    i
%     [detectionsAboveThreshold{i}, detectionsAtThreshold{i} bestRootLoc{i} bestPartLoc{i} bestRootLevel{i} bestComponentIdx{i} bestScore{i}] = detect(feats{i}, scales, model, threshold);
     [detectionsAboveThreshold{i}, detectionsAtThreshold{i} bestRootLoc{i} bestPartLoc{i} bestRootLevel{i} bestComponentIdx{i} bestScore{i}] = detect(dummy, scales, model, threshold);
     break;
end
detectTime = toc(dtic)
%matlabpool close;
%{
detectionsAboveThreshold = detectionsAboveThreshold(idxs);
detectionsAtThreshold = detectionsAtThreshold(idxs);
bestRootLoc = bestRootLoc(idxs);
bestPartLoc = bestPartLoc(idxs);
bestRootLevel = bestRootLevel(idxs);
bestComponentIdx = bestComponentIdx(idxs);
bestScore = bestScore(idxs);
%}

detectionsAboveThreshold(1:n) = detectionsAboveThreshold(1);
detectionsAtThreshold(1:n) = detectionsAtThreshold(1);
bestRootLoc(1:n) = bestRootLoc(1);
bestPartLoc(1:n) = bestPartLoc(1);
bestRootLevel(1:n) = bestRootLevel(1);
bestComponentIdx(1:n) = bestComponentIdx(1);
bestScore(1:n) = bestScore(1);

end

