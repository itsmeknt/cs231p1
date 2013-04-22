function [neg] = updateNegCache(allNeg, oldNeg, model)

neg = [];
ambiguousData = [];
for i=1:length(allNeg)
    im = allNeg(i).im;
    id = allNeg(i).id;
    [feat scale] = featpyramid(im, model.sbin, model.interval);
    [dummy, ambiguous, negHard] = detect(feat, scale, model, 1, false);
    for j=1:length(negHard)
        entry.im = im;
        entry.id = id;
        bbox = negHard(i).rootBbox;
        entry.x1 = bbox(1);
        entry.y1 = bbox(2);
        entry.x2 = bbox(3);
        entry.y2 = bbox(4);
        neg = [neg; entry];
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
    if oldIdSet.contains(ambiguousData(i).id)
        neg = [neg; ambiguousData(i)];
    end
end