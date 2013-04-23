function [neg] = updateNegCache(allNeg, oldNeg, model, sizeLimit)

neg = [];
ambiguousData = [];
for i=1:length(allNeg)
    if length(neg) > sizeLimit
        break;
    end
    [feat scale] = loadFeaturePyramidCache(allNeg(i).id);
    [dummy, ambiguous, negHard bestRootLoc bestPartLoc bestRootLevel bestComponentIdx bestScore] = detect(feat, scale, model, 1);
    if bestScore < 1
        continue;
    end
    entry.bestRootLoc = bestRootLoc;
    entry.bestPartLoc = bestPartLoc;
    entry.bestRootLevel = bestRootLevel;
    entry.bestComponentIdx = bestComponentIdx;
    entry.im = allNeg(i).im;
    entry.id = allNeg(i).id;
    neg = [neg; entry];
    if length(neg) > sizeLimit
        break;
    end
    
    for j=1:length(ambiguous)
        entry.im = im;
        entry.id = id;
        bbox = ambiguous(i).rootBbox;
        entry.x1 = bbox(1);
        entry.y1 = bbox(2);
        entry.x2 = bbox(3);
        entry.y2 = bbox(4);
        ambiguousData = [ambiguousData; entry];
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
        neg = [neg; ambiguousData(i)];
        if length(neg) > sizeLimit
            break;
        end
    end
end