function [neg] = updateNegCache(allNeg, oldNeg, model, sizeLimit)

neg = [];
ambiguousData = [];
for i=1:length(allNeg)
    if length(neg) > sizeLimit
        break;
    end
    im=imread(allNeg(i).im);
    im = color(im);
    id = allNeg(i).id;
    [feat scale] = featpyramid(im, model.sbin, model.interval);
    [dummy, ambiguous, negHard] = detect(feat, scale, model, 1, false);
    maxScore = -realmax;
    hardestNeg = [];
    for j=1:length(negHard)
        if negHard(i).score > maxScore
            maxScore = negHard(i).score;
            hardestNeg = [negHard(i)];
        elseif negHard(i).score == maxScore
            hardestNeg = [hardestNeg; negHard(i)];
        end
    end
    for j=1:length(hardestNeg)
        entry.im = im;
        entry.id = id;
        bbox = hardestNeg(i).rootBbox;
        entry.x1 = bbox(1);
        entry.y1 = bbox(2);
        entry.x2 = bbox(3);
        entry.y2 = bbox(4);
        neg = [neg; entry];
        if length(neg) > sizeLimit
            break;
        end
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