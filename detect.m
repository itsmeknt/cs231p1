function [boxes] = detect(input, model, thresh)

% boxes = detect(input, model, thresh)
% Detect objects in input using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% The first 4 columns specify the bounding box for the root filter and
% subsequent columns specify the bounding boxes of each part.
%
% If bbox is not empty, we pick best detection with significant overlap. 
% If label and fid are included, we write feature vectors to a data file.


% NOTE: You'll need to implement the inference for the part filters in this
% file

% we assume color images
input = color(input);

% the feature pyramid
[feat, pyramidScales] = featpyramid(input, model.sbin, model.interval);

% score
[scores padx pady] = computeScores(feat, model, true);

% find all good bounding boxes
boxes = [];
for componentIdx=1:model.numcomponents
    for pyramidLevelIdx=1:length(pyramidScales)
        % get all good matches
        scale = pyramidScales(pyramidLevelIdx);
        rsize = model.rootfilters{model.components{componentIdx}.rootindex}.size;
        
        score = scores{componentIdx}{pyramidLevelIdx};
        I = find(score > thresh);
        [Y, X] = ind2sub(size(score), I);
        boxesEntry = zeros(length(I), 6);
        for i = 1:length(I)
            x = X(i);
            y = Y(i);
            b = getBoundingBox(x, y, scale, padx, pady, rsize);
            boxesEntry(i,:) = [b componentIdx score(I(i))];
        end
        boxes = [boxes; boxesEntry];
    end
end
end




%{
old code

[scores scales padx pady] = computeScores(input, model, true);
boxes = [];
for componentIdx=1:length(scores)
    for pyramidLevelIdx=1:length(scores{componentIdx})
        % get all good matches
        scale = scales(pyramidLevelIdx);
        rsize = model.rootfilters{model.components{componentIdx}.rootindex}.size;
        
        score = scores{componentIdx}{pyramidLevelIdx};
        I = find(score > thresh);
        [Y, X] = ind2sub(size(score), I);
        boxesEntry = zeros(length(I), 6);
        for i = 1:length(I)
            x = X(i);
            y = Y(i);
            b = getBoundingBox(x, y, scale, padx, pady, rsize);
            boxesEntry(i,:) = [b componentIdx score(I(i))];
        end
        boxes = [boxes; boxesEntry];
    end
end
%}