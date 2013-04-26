function [detections detectionsAtThreshold] = detect(featPyramid, scales, model, threshold, latent)

% detects objects in feat pyramid
%
% inputs:
% featPyramid - feature pyramid from featpyramid.m
% scales - scales of the feature pyramid from featpyramid.m
% model - model to use for detection
% threshold - detection threshold
% latent - if true, detections returns only the best
%       detection. If false, detections returns all detections above
%       threshold

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

bestScore = -inf;
bestDetection = [];
detectionsAboveThreshold = [];
detectionsAtThreshold = [];
for pLevelIdx = 1:length(scales)
    scale = scales(pLevelIdx);
    if size(featPyramid{level}, 1)+2*pady < model.maxsize(1) || size(featPyramid{level}, 2)+2*padx < model.maxsize(2)
        continue;
    end
    
    for cIdx = 1:model.numcomponents
        rootSize = model.rootfilters{model.components{cIdx}.rootindex}.size;
        
        score = Array{cIdx, pLevelIdx};
       
        % threshold scores
        Iabove = find(score > threshold);
        detectionsAboveThreshold = [detectionsAboveThreshold; formatDetections(score, Iabove, cIdx, scale, model.padx, model.pady, rootSize)];
        Iat = find(score == threshold);
        detectionsAtThreshold = [detectionsAtThreshold; formatDetections(score, Iat, cIdx, scale, model.padx, model.pady, rootSize)];
    end
end
end

ScoreMatrix=cell(model.numcomponents,length(features));

% Find the original scale and last scale indices
orig_scale=find(scales==1);
last_scale=size(scales,1);
numScales = last_scale-orig_scale+1;

% initialize fconv variables
rootfilters = cell(1, length(model.rootfilters));
for i=1:length(model.rootfilters)
    rootfilters{i} = model.rootfilters{i}.w;
end
partfilters = cell(1, length(model.partfilters));
for i=1:length(model.partfilters)
    partfilters{i} = model.partfilters{i}.w;
end

maxScore = -realmax;

% fconv after padding
conv_roots = cell(length(scales), length(model.rootfilters));
conv_parts = cell(length(scales), length(model.partfilters));
for k=1:length(scales)
    featuresPadded = padarray(features{k}, [model.pady model.padx 0], 0);
    
    if k > model.interval
        conv_roots(k, :) = fconv(featuresPadded, rootfilters, 1, length(rootfilters));    
    end
    if k <= length(scales) - model.interval
        conv_parts(k, :) = fconv(featuresPadded, partfilters, 1, length(partfilters));
    end
end

for i=1:model.numcomponents
    rootindex = model.components{i}.rootindex;
    rootsize = model.rootfilters{rootindex}.size;
    
    % initialize deformation matrix
    deform=zeros(2*rootsize(1),2*rootsize(2),model.numparts);       % tensor vector
    for j=1:model.numparts
        % Compute the deformation cost matrix
        defIdx=model.components{i}.parts{j}.defindex;
        partDef=model.defs{defIdx};
        deform(:,:,j)=computeDefMatrix(rootsize,partDef);
    end 
    
    % Compute the score for each location of the root
    % parallelize
    %matlabpool open feature('numcores');
    for k=orig_scale:last_scale
        rootScoreMatrix = conv_roots{k, model.components{i}.rootindex};
        if size(rootScoreMatrix,1)-2*model.pady <= 0 || size(rootScoreMatrix,2)-2*model.padx <= 0
            ScoreMatrix{i, k} = zeros(0,0);
            continue;
        end
        
        ScoreMatrix{i, k} = rootScoreMatrix(1+model.pady:end-model.pady,1+model.padx:end-model.padx) + model.offsets{rootindex}.w;
        
        convPartSize = size(conv_parts{k-model.interval, 1});
        partConvTensors=zeros(convPartSize(1), convPartSize(2), model.numparts);
        for j=1:model.numparts
            partindex = model.components{i}.parts{j}.partindex;
            partConvTensors(:,:,j) = conv_parts{k-model.interval, partindex};
        end
        
        for x=1:size(features{k},2)-rootsize(2)+1
            for y=1:size(features{k},1)-rootsize(1)+1
                partScores = partConvTensors((2*y):(2*y+2*rootsize(1)-1),(2*x):(2*x+2*rootsize(2)-1),:) + deform;
                
                [max_xs,ind_xs]=max(partScores, [], 2);
                [bestPartScores,partLocsY]=max(max_xs, [], 1);
                
                partLocs = zeros(model.numparts, 2);
                for p=1:model.numparts
                    partLocs(p,1) = partLocsY(1,1,p);
                    partLocs(p,2) = ind_xs(partLocsY(1,1,p), 1, p);
                end
                ScoreMatrix{i,k}(y,x)=ScoreMatrix{i,k}(y,x)+sum(bestPartScores);
                
                if (ScoreMatrix{i,k}(y,x)>maxScore)
                    maxScore=ScoreMatrix{i,k}(y,x);
                    component=i;
                    rootLoc=[y x];
                    partLoc=partLocs;
                    level=k;
                end
            end
        end
    end
end


function detections = formatDetections(score, I, componentIdx, scale, padx, pady, rootSize)
detections = [];
[Y, X] = ind2sub(size(score), I);
for i = 1:length(I)
    x = X(i);
    y = Y(i);
    rootBbox = getBoundingBox(x, y, scale, padx, pady, rootSize);
    entry.rootBbox = rootBbox;
    entry.partBbox = [];
    entry.score = score(I(i));
    entry.component = componentIdx;
    detections = [detections; entry];
end
end

