function [scores, scales, padx, pady] = computeScores(input, model, addDeformationCost)

% Computes scores of each block of the image at each pyramid level.
%
% Returns scores, which is a cell array with model.numcomponents number of
% cells.
%
% scores{component} is a cell array with model.interval number of cells.
% scores{component, 1} gives the scores at the feature pyramid level at the original
% resolution,  and scores{component, interval} give the scores at the feature
% pyramid level at (1/2^{(interval-1)/interval}) resolution.
%
% scores{component}{pyramidLevel} = mxn matrix where m = number of blocks
% along x axis and n = number of blocks along y axis for the given pyramid
% scale.
%
% Returns scales, which is the scale of the feature pyramid level. Same
% for all model components
%
% Returns padx, which is the length of the padding of the feature on one side along x axis
% Returns pady, which is the length of the padding of the feature on one side along y axis
%
% You need to save scales, padx, and pady to compute the bounding box of
% scores
%
% NOTE: You'll need to implement the inference for the part filters in this
% file

% we assume color images
input = color(input);

% prepare model for convolutions
rootfilters = cell(length(model.rootfilters), 1);
for i = 1:length(model.rootfilters)
  rootfilters{i} = model.rootfilters{i}.w;
end

% cache some data
ridx = cell(model.numcomponents, 1);
oidx = cell(model.numcomponents, 1);
root = cell(model.numcomponents, 1);
for c = 1:model.numcomponents
  ridx{c} = model.components{c}.rootindex;
  oidx{c} = model.components{c}.offsetindex;
  root{c} = model.rootfilters{ridx{c}}.w;
end

% we pad the feature maps to detect partially visible objects
padx = ceil(model.maxsize(2)/2+1);
pady = ceil(model.maxsize(1)/2+1);

% the feature pyramid
interval = model.interval;
[feat, pyramidScales] = featpyramid(input, model.sbin, interval);

% initialize return variables
levelIdx = 1;
scores = cell(model.numcomponents, 1);
for i = 1:model.numcomponents
    scores{1} = cell(length(feat)-interval, 1);
end
scales = zeros(length(feat)-interval, 1);

% compute score at each scale
for level = interval+1:length(feat)
  scale = model.sbin/pyramidScales(level);    
  if size(feat{level}, 1)+2*pady < model.maxsize(1) || ...
     size(feat{level}, 2)+2*padx < model.maxsize(2)
    continue;
  end
    
  % convolve feature maps with filters 
  featr = padarray(feat{level}, [pady padx 0], 0);
  rootmatch = fconv(featr, rootfilters, 1, length(rootfilters));
 
  for c = 1:model.numcomponents
    % root score + offset
    scoreEntry = rootmatch{ridx{c}} + model.offsets{oidx{c}}.w;  
    scores{c}{levelIdx} = scoreEntry;
  end
  scales(levelIdx) = scale;
  levelIdx = levelIdx+1;
end
