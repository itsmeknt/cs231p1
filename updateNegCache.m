function [neg] = updateNegCache(allNeg, oldNeg, model, sizeLimit)
[bboxes bestDetections detectionsAtThresholds] = detectParallel(allNeg, model, -1, true, false);
dummy.bestRootLoc = zeros(1,2);
dummy.bestPartLoc = zeros(model.numparts, 2);
dummy.bestRootLevel = 0;
dummy.bestComponentIdx = 0;
dummy.im = 'dummy';
dummy.id = 0;
n = min(length(allNeg), sizeLimit);
neg(1:n) = dummy;
ambiguousData(1:n) = dummy;
negIdx = 1;
r_idx = randperm(length(allNeg));
for idx=1:length(r_idx)
    i = r_idx(idx);
    if negIdx > sizeLimit
        break;
    end
    
    bestDetection = bestDetections{idx};
    if bestDetection.score < -1
        continue;
    end
    
    entry.bestRootLoc = bestDetection.rootLoc;
    entry.bestPartLoc = bestDetection.partLocs;
    entry.bestRootLevel = bestDetection.level;
    entry.bestComponentIdx = bestDetection.component;
    entry.im = allNeg(i).im;
    entry.id = allNeg(i).id;
    neg(negIdx) = entry;
    negIdx = negIdx+1;
    if negIdx > sizeLimit
        break;
    end
    
    for j=1:length(detectionsAtThresholds{i})
        entry.im = allNeg(i).im;
        entry.id = allNeg(i).id;
        bbox = detectionsAtThresholds{i}(j).rootBbox;
        entry.x1 = bbox(1);
        entry.y1 = bbox(2);
        entry.x2 = bbox(3);
        entry.y2 = bbox(4);
        ambiguousData(i) = entry;
    end
end


% keep old negs that are ambiguous
oldIdSet = java.util.HashSet;
for i=1:length(oldNeg)
    id = oldNeg(i).id;
    oldIdSet.add(id);
end

for i=1:length(ambiguousData)
    if negIdx > sizeLimit
        break;
    end
    if oldIdSet.contains(ambiguousData(i).id)
        neg(negIdx) = ambiguousData(i);
        negIdx = negIdx+1;
        if negIdx > sizeLimit
            break;
        end
    end
end
neg = neg(1:negIdx-1);            % truncate off empty elements at the end