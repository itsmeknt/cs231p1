function [neg] = updateNegCache(allNeg, oldNeg, model, sizeLimit)
[bboxes bestDetections detectionsAtThresholds] = detectParallel(allNeg, model, -1, true, false);

neg = [];
ambiguousData = [];
r_idx = randperm(length(allNeg));
for idx=1:length(r_idx)
    i = r_idx(idx);
    if length(neg) > sizeLimit
        break;
    end
    
    bestDetection = bestDetections{i};
    if bestDetection.score < -1
        continue;
    end
    
    entry.bestRootLoc = bestDetection(1).rootLoc;
    entry.bestPartLoc = bestDetection(1).partLocs;
    entry.bestRootLevel = bestDetection(1).level;
    entry.bestComponentIdx = bestDetection(1).component;
    entry.im = allNeg(i).im;
    entry.id = allNeg(i).id;
    entry.x1 = bestDetection.rootBbox(1);
    entry.y1 = bestDetection.rootBbox(2);
    entry.x2 = bestDetection.rootBbox(3);
    entry.y2 = bestDetection.rootBbox(4);
    neg = [neg; entry];
    if length(neg) > sizeLimit
        break;
    end
    
    for j=1:length(detectionsAtThresholds{i})
        entry.im = allNeg(i).im;
        entry.id = allNeg(i).id;
        dAtThresh = detectionsAtThresholds{i};
        bbox = dAtThresh(j).rootBbox;
        entry.bestRootLoc = dAtThresh(j).rootLoc;
        entry.bestPartLoc = dAtThresh(j).partLocs;
        entry.bestRootLevel = dAtThresh(j).level;
        entry.bestComponentIdx = dAtThresh(j).component;
        entry.x1 = bbox(1);
        entry.y1 = bbox(2);
        entry.x2 = bbox(3);
        entry.y2 = bbox(4);
        ambiguousData = [ambiguousData;entry];
    end
end


% keep old negs that are ambiguous
oldIdSet = java.util.HashSet;
for i=1:length(oldNeg)
    id = oldNeg(i).id;
    oldIdSet.add(id);
end

for i=1:length(ambiguousData)
    if length(neg) > sizeLimit
        break;
    end
    if oldIdSet.contains(ambiguousData(i).id)
        neg = [neg;ambiguousData(i)];
        if length(neg) > sizeLimit
            break;
        end
    end
end