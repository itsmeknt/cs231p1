function [detectionBboxes detections, detectionsAtThreshold] = detectParallel(posOrNegs, model, threshold, chooseBest, doPosLatent)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%[dummy, scales] = loadFeaturePyramidCache(posOrNegs(1).id, posOrNegs(1).im, model.sbin, model.interval);
n = length(posOrNegs);
%feats = cell(1,n);
%idxs = 1:n;
%matlabpool open;
%parfor p=1:n
%    feats{p} = loadFeaturePyramidCache(posOrNegs(p).id);
%end

sbin = model.sbin;
interval = model.interval;
detectionBboxes = cell(1,n);
detections = cell(1,n);
detectionsAtThreshold = cell(1,n);
dtic = tic;
for i=1:n
    if (doPosLatent)
        entry = posOrNegs(i);
        trueBbox = [entry.x1, entry.y1, entry.x2, entry.y2];
    else
        trueBbox = [];
    end
    id = posOrNegs(i).id
    [feat, scales] = loadFeaturePyramidCache(posOrNegs(i).id, posOrNegs(i).im, sbin, interval);
    [detectionBboxes{i} detections{i}, detectionsAtThreshold{i}] = detect(feat, scales, model, threshold, chooseBest, trueBbox);
%     [detectionBboxes{i} detections{i}, detectionsAtThreshold{i}] = detect(dummy, scales, model, threshold, chooseBest, trueBbox);
%     break;
end
detectTime = toc(dtic)
%matlabpool close;

%detectionBboxes = detectionBboxes(idxs);
%detections = detections(idxs);
%detectionsAtThreshold = detectionsAtThreshold(idxs);
end

