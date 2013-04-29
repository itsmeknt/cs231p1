function [detectionBboxes detections detectionsAtThreshold] = detect(featPyramid, scales, model, threshold, chooseBest, trueBbox)

% detects objects in feat pyramid
%
% inputs:
% featPyramid - feature pyramid from featpyramid.m
% scales - scales of the feature pyramid from featpyramid.m
% model - model to use for detection
% threshold - detection threshold
% chooseBest - if true, detections returns only the best detections (usually one, unless there is a tie in score).
%       If false, detections returns all detections above threshold
%       For positive latent, should set this to true and provide trueBbox.
% trueBbox - The true bbox annotated by the data.

% output:
% detections - array of detection structs. Has to be above threshold.
% detectionsAtThreshold - array of detection structs at threshold.
% detection structs have the
% following member variables:
%   .rootBbox - bounding box ([x1, y1, x2, y2]) of root position
%   .partBbox - a matrix of bounding box where each row represents the bounding
%               box of a part. So it is a px4 matrix. The index for the
%               part in this matrix is thes ame as the index in
%               model.filters.
%   .score - score of the detection
%   .component - the component used for the detection
%   .level - pyramid level of root filter
%   .rootLoc - [x,y] of root location in feature
%                                       space
%   .partLocs - a px2 matrix where each row is a
%                                   [x,y] of part location in feature space

% prepare model for convolutions
rootfilters = cell(length(model.rootfilters), 1);
for i = 1:length(model.rootfilters)
  rootfilters{i} = model.rootfilters{i}.w;
end
partfilters = cell(length(model.partfilters), 1);
for i = 1:length(model.partfilters)
  partfilters{i} = model.partfilters{i}.w;
end

% cache some data
ridx = cell(model.numcomponents, 1);
oidx = cell(model.numcomponents, 1);
root = cell(model.numcomponents, 1);
rsize = cell(model.numcomponents, 1);
numparts = cell(model.numcomponents, 1);
pidx = cell(model.numcomponents, model.numparts);
didx = cell(model.numcomponents, model.numparts);
part = cell(model.numcomponents, model.numparts);
psize = cell(model.numcomponents, model.numparts);
rpidx = cell(model.numcomponents, model.numparts);
for c = 1:model.numcomponents
  ridx{c} = model.components{c}.rootindex;
  oidx{c} = model.components{c}.offsetindex;
  root{c} = model.rootfilters{ridx{c}}.w;
  rsize{c} = [size(root{c},1) size(root{c},2)];
  numparts{c} = length(model.components{c}.parts);
  for j = 1:model.numparts
    pidx{c,j} = model.components{c}.parts{j}.partindex;
    didx{c,j} = model.components{c}.parts{j}.defindex;
    part{c,j} = model.partfilters{pidx{c,j}}.w;
    psize{c,j} = [size(part{c,j},1) size(part{c,j},2)];
    % reverse map from partfilter index to (component, part#)
    rpidx{pidx{c,j}} = [c j];
  end
end

padx = model.padx;
pady = model.pady;

detections = [];
detectionsAtThreshold = [];
maxOverlap = -inf;              % only used for latent root filter detection
for pLevelIdx = model.interval+1:length(scales)
    scale = model.sbin/scales(pLevelIdx);
    
    % skip sizes too small
    if size(featPyramid{pLevelIdx}, 1)+2*pady < model.maxsize(1) || size(featPyramid{pLevelIdx}, 2)+2*padx < model.maxsize(2)
        continue;
    end
    
    % convolve feature maps with filters
    featr = padarray(featPyramid{pLevelIdx}, [pady padx 0], 0);
    conv_roots = fconv(featr, rootfilters, 1, length(rootfilters));
    
    %{
    if ~isempty(partfilters)
        featp = padarray(featPyramid{pLevelIdx-model.interval}, [2*pady 2*padx 0], 0);
        conv_parts = fconv(featp, partfilters, 1, length(partfilters));
    end
  %}
    
    for cIdx = 1:model.numcomponents
        rootSize = model.rootfilters{model.components{cIdx}.rootindex}.size;
        partSizes = zeros(model.numparts, 2);
        for p=1:model.numparts
            w = model.partfilters{model.components{cIdx}.parts{p}.partindex}.w;
            partSizes(p,:) = [size(w, 1), size(w, 2)];
        end
        rootScoreMatrix = conv_roots{model.components{cIdx}.rootindex};
        rootConvSize = size(rootScoreMatrix);
        
        % root score + offset
        score = conv_roots{ridx{c}} + model.offsets{oidx{c}}.w;
        
        %{
        transform_scores = cell(1,6);
        transform_positions = cell(1,6);
        for p = 1:model.numparts
            [transform_scores{p} transform_positions{p}] = TwoDDistTransform(conv_parts{pidx{cIdx, p}}, model.defs{model.components{cIdx}.parts{p}.defindex}.w);
        end
        
        %start
        %{
        partLocs = cell(rootConvSize(1), rootConvSize(2));
        for rConvX=1:rootConvSize(2)
            for rConvY=1:rootConvSize(1)
                bestPartScores = zeros(model.numparts, 1);
                partLocs{rConvY, rConvX} = zeros(model.numparts, 2);
                for p=1:model.numparts
                    anchor = model.defs{model.components{cIdx}.parts{p}.defindex}.anchor;
                    anchorAbsolute = 2*([rConvY rConvX]) + [anchor(2) anchor(1)];
                    transform_score_matrix = transform_scores{p};
                    bestPartScores(p) = transform_score_matrix(anchorAbsolute(1), anchorAbsolute(2));
                    partLocs{rConvY, rConvX}(p,:) = [transform_positions{p}(anchorAbsolute(1), anchorAbsolute(2), 1), transform_positions{p}(anchorAbsolute(1), anchorAbsolute(2), 2)];
                end
                score(rConvY, rConvX) = score(rConvY, rConvX) + sum(bestPartScores);
            end
        end
        %}
                %end
                
            % start
    
        % initialize deformation tensor
        deform=cell(1, model.numparts);       % tensor vector
        for j=1:model.numparts
            % Compute the deformation cost matrix
            partSize = size(model.partfilters{pidx{cIdx, p}}.w);
            
            defIdx=model.components{cIdx}.parts{j}.defindex;
            partDef=model.defs{defIdx};
            deform{j}=computeDefMatrix(rootSize,partSize,partDef);
        end
        
        % part score + deformation cost
        for rConvX=1:rootConvSize(2)
            for rConvY=1:rootConvSize(1)
                partScores = 0;
                for p=1:model.numparts
                    
                    
                end
                
                    
                bestPartScores = zeros(model.numparts, 1);
                partLocs = zeros(model.numparts, 2);
                
                for p = 1:model.numparts
                    partSize = size(model.partfilters{pidx{cIdx, p}}.w);
                    rFeatXmin = rConvX;
                    rFeatYmin = rConvY;
                    
                    pFeatXmin = 2*(rFeatXmin-1)+1;
                    pFeatXmax = pFeatXmin + 2*rootSize(2) - 1;
                    pFeatYmin = 2*(rFeatYmin-1)+1;
                    pFeatYmax = pFeatYmin + 2*rootSize(1) - 1;
                    
                    pConvXmin = pFeatXmin;
                    pConvXmax = pFeatXmax - partSize(2) + 1;
                    pConvYmin = pFeatYmin;
                    pConvYmax = pFeatYmax - partSize(1) + 1;
                    partScores = conv_parts{pidx{cIdx, p}}(pConvYmin:pConvYmax, pConvXmin:pConvXmax) - deform{p};
                    
                    
                    
                    [maxCol, yIdx] = max(partScores);
                    [bestPartScores(p) xIdx] = max(maxCol);
                    partLocs(p,:) = [yIdx(xIdx), xIdx];
                end
            
                % end
                
                score(rConvY,rConvX)=score(rConvY,rConvX)+sum(bestPartScores);
                
                if chooseBest
                    add = false;
                    if isempty(detections)
                        add = true;
                    else
                        if (~isempty(trueBbox) && computeOverlap(trueBbox, [rConvX, rConvY, rConvX + rootSize(2) - 1, rConvY + rootSize(1) - 1]) > 0.7)
                            add = true;
                        elseif (score(rConvY,rConvX)>detections(1).score)
                            add = true;
                        end
                    end
                    
                    if (add)
                        detections = formatDetection(score(rConvY,rConvX), [rConvY,rConvX], partLocs, cIdx, scale, pLevelIdx, model.padx, model.pady, rootSize, partSizes);
                    end
                end
            end
        end
        %}
        partLocs = ones(model.numparts, 2);
        if ~chooseBest
            Iabove = find(score > threshold);
            detections = [detections; formatDetections(score, Iabove, partLocs, cIdx, scale, pLevelIdx, model.padx, model.pady, rootSize, partSizes, model.numparts, model.sbin)];
        else
            if isempty(trueBbox)
                [y_max y_ind] = max(score);
                [maxScore x_ind] = max(y_max);
                x = x_ind;
                y = y_ind(x_ind);
            else
                trueBboxInScoreSpace = [model.padx model.pady model.padx model.pady] + ceil(trueBbox/scale) + [0 0 -rootSize(2)+1 -rootSize(1)+1];        % since score space is convolution space, we are only interested in the top left corner
                trueBboxInScoreSpace(1) = min(size(score,2), trueBboxInScoreSpace(1));
                trueBboxInScoreSpace(2) = min(size(score,1), trueBboxInScoreSpace(2));
                trueBboxInScoreSpace(3) = max(trueBboxInScoreSpace(1), trueBboxInScoreSpace(3));
                trueBboxInScoreSpace(4) = max(trueBboxInScoreSpace(2), trueBboxInScoreSpace(4));
                minX = max(1, trueBboxInScoreSpace(1)-floor(rootSize(2)*0.3));
                maxX = min(size(score,2), trueBboxInScoreSpace(3)+floor(rootSize(2)*0.3));
                minY = max(1, trueBboxInScoreSpace(2)-floor(rootSize(1)*0.3));
                maxY = min(size(score,1), trueBboxInScoreSpace(4)+floor(rootSize(1)*0.3));

                
                clippedScore = score(minY:maxY, minX:maxX);
               
                x = -1;
                y = -1;
                maxScore = -inf;
                loop = true;
                maxOverlapX = -1;
                maxOverlapY = -1;
                maxOverlapScore= -inf;
                while loop && max(max(clippedScore)) ~= -inf
                    loop = false;
                    [y_max_clipped y_ind_clipped] = max(clippedScore);
                    [maxScore x_ind_clipped] = max(y_max_clipped);
                    x = x_ind_clipped;
                    y = y_ind_clipped(x_ind_clipped);
                    
                    predBbox = getBoundingBox(x, y, scale, model.padx, model.pady, rootSize);
                    overlap = computeOverlap(predBbox, trueBbox);
                    if overlap < 0.7
                        clippedScore(y, x) = -inf;
                        loop = true;
                    end
                    if overlap > maxOverlap
                        maxOverlap = overlap;
                        maxOverlapX = x;
                        maxOverlapY = y;
                        maxOverlapScore = maxScore;
                    end
                        
                end
            end
            
            
            if isempty(trueBbox) && (isempty(detections) || maxScore > detections(1).score)
                detections = formatDetection(maxScore, [y,x], partLocs, cIdx, scale, pLevelIdx, model.padx, model.pady, rootSize, partSizes, model.sbin);
            elseif ~isempty(trueBbox)
                if maxOverlap > 0.7
                    detections = formatDetection(maxScore, [y,x], partLocs, cIdx, scale, pLevelIdx, model.padx, model.pady, rootSize, partSizes, model.sbin);
                elseif maxOverlapScore ~= -inf
                    detections = formatDetection(maxOverlapScore, [maxOverlapY,maxOverlapX], partLocs, cIdx, scale, pLevelIdx, model.padx, model.pady, rootSize, partSizes, model.sbin);
                end
            end
        end
        
        Iat = find(score == threshold);
        detectionsAtThreshold = [detectionsAtThreshold; formatDetections(score, Iat, partLocs, cIdx, scale, pLevelIdx, model.padx, model.pady, rootSize, partSizes, model.numparts, model.sbin)];
    end
end

detectionBboxes = zeros(size(detections, 1), 6);
for i=1:size(detections,1)
    detectionBboxes(i,:) = [detections(i).rootBbox detections(i).component detections(i).score];
end
end



function detections = formatDetections(score, I, partLocs, componentIdx, scale, pyramidLevel, padx, pady, rootSize, partSizes, numparts, sbin)
detections = [];
[Y, X] = ind2sub(size(score), I);
for i = 1:length(I)
    x = X(i);
    y = Y(i);
    entry = formatDetection(score(y,x), [y, x], partLocs, componentIdx, scale, pyramidLevel, padx, pady, rootSize, partSizes, sbin);
    detections = [detections; entry];
end
end

function detection = formatDetection(score, rootLoc, partLocs, componentIdx, scale, pyramidLevel, padx, pady, rootSize, partSizes, sbin)
    rootBbox = getBoundingBox(rootLoc(2), rootLoc(1), scale, padx, pady, rootSize);
    detection.rootBbox = rootBbox;
    detection.partBbox = zeros(size(partLocs,1), 4);
    for i=1:size(partLocs,1)
        partBbox = getBoundingBox(2*rootLoc(2) + partLocs(i,2), 2*rootLoc(1) + partLocs(i,1), 2*scale, 2*padx, 2*pady, partSizes(i,:));
        detection.partBbox(i,:) = partBbox;
    end
    detection.score = score;
    detection.component = componentIdx;
    detection.level = pyramidLevel;
    detection.rootLoc = rootLoc; 
    detection.partLocs = partLocs;   
end


function dummy = getDummyDetectionStruct(numparts)                    
dummy.rootBbox = zeros(1, 4) - 1;
dummy.partBbox = zeros(numparts, 4) - 1;
dummy.score = -1;
dummy.component = -1;
dummy.level = -1;
dummy.rootLoc = zeros(1, 2) - 1;
dummy.partLoc = zeros(numparts, 2) - 1;
end

