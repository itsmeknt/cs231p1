function [ newPos] = updateLatentRootPosition( pos, model )
%UPDATELATENTPOSITION Summary of this function goes here
%   Detailed explanation goes here
 newPos = pos;
[bboxes bestDetections ambiguousCases] = detectParallel(newPos, model, 0, true, true);
for i=1:length(newPos)
    bestDetection = bestDetections{i};
    newPos(i).bestRootLoc = [model.pady model.padx] + ceil([newPos(i).y1 newPos(i).x1]/model.sbin); %bestDetection(1).rootLoc;
    newPos(i).bestPartLoc = bestDetection(1).partLocs;
    newPos(i).bestRootLevel = 11; %bestDetection(1).level;
    newPos(i).bestComponentIdx = bestDetection(1).component;
end
end

