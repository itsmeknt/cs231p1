function model = initmodel(pos,varargin)

% model = initmodel(pos, sbin, size)
% Initialize model structure.
%
% If not supplied the dimensions of the model template are computed
% from statistics in the postive examples.
% 
% This should be documented! :-)
% model.sbin
% model.interval
% model.numblocks
% model.numcomponents
% model.blocksizes
% model.regmult
% model.learnmult
% model.maxsize
% model.minsize
% model.padx
% model.pady
% model.rootfilters{i}
%   .size
%   .w
%   .blocklabel
% model.partfilters{i}
%   .w
%   .blocklabel
% model.defs{i}
%   .anchor
%   .w
%   .blocklabel
% model.offsets{i}
%   .w
%   .blocklabel
% model.components{i}
%   .rootindex
%   .parts{j}
%     .partindex
%     .defindex
%   .offsetindex
%   .dim
%   .numblocks

% pick mode of aspect ratios
h = [pos(:).y2]' - [pos(:).y1]' + 1;
w = [pos(:).x2]' - [pos(:).x1]' + 1;
xx = -2:.02:2;
filter = exp(-[-100:100].^2/400);
aspects = hist(log(h./w), xx);
aspects = convn(aspects, filter, 'same');
[peak, I] = max(aspects);
aspect = exp(xx(I));

% pick 20 percentile area
areas = sort(h.*w);
area = areas(floor(length(areas) * 0.2));
area = max(min(area, 5000), 3000);

% pick dimensions
w = sqrt(area/aspect);
h = w*aspect;

% size of HOG features
if nargin < 4
  model.sbin = 8;
else
  model.sbin = sbin;
end

% size of root filter
if nargin < 5
  model.rootfilters{1}.size = [round(h/model.sbin) round(w/model.sbin)];
else
  model.rootfilters{1}.size = size;
end

% set up offset 
model.offsets{1}.w = 0;
model.offsets{1}.blocklabel = 1;
model.blocksizes(1) = 1;
model.regmult(1) = 0;
model.learnmult(1) = 20;
model.lowerbounds{1} = -100;

% set up root filter
model.rootfilters{1}.w = zeros([model.rootfilters{1}.size 31]);
height = model.rootfilters{1}.size(1);
% root filter is symmetric
width = ceil(model.rootfilters{1}.size(2)/2);
model.rootfilters{1}.blocklabel = 2;
model.blocksizes(2) = width * height * 31;
model.regmult(2) = 1;
model.learnmult(2) = 1;
model.lowerbounds{2} = -100*ones(model.blocksizes(2),1);

% set up parts and deformations
% part filter is symmetric
model.numparts = 6;
numcomponents = 1;
rHeight = model.rootfilters{1}.size(1);
rWidth = model.rootfilters{1}.size(2);
partArea = 0.8*rHeight*rWidth;
pWidth = ceil(sqrt(partArea*(rWidth/rHeight)));
pHeight = ceil(pWidth*(rHeight/rWidth));

blockLabel = 2;
cumIdx = 0;
for componentIdx = 1:numcomponents
    for partIdx = 1:model.numparts
        cumIdx = cumIdx+1;
        blockLabel = blockLabel+1;
        model.partfilters{cumIdx}.blockLabel = blockLabel;
        model.partfilters{cumIdx}.w = zeros(pHeight, pWidth, 31);
        model.blocksizes(blockLabel) = prod(size(model.partfilters{cumIdx}.w));
        model.regmult(blockLabel) = 1;
        model.learnmult(blockLabel) = 1;
        model.lowerbounds{blockLabel} = -100*ones(model.blocksizes(blockLabel),1);
        
        blockLabel = blockLabel+1;
        model.defs{cumIdx}.blockLabel = blockLabel;
        model.defs{cumIdx}.w = [0, 0, 1, 1];
        model.blocksizes(blockLabel) = prod(size(model.defs{cumIdx}.w));
        model.regmult(blockLabel) = 1;
        model.learnmult(blockLabel) = 1;
        model.lowerbounds{blockLabel} = [-100, -100, 0, 0];
        
        model.components{componentIdx}.parts{partIdx}.partindex = cumIdx;
        model.components{componentIdx}.parts{partIdx}.partidx = cumIdx;
        model.components{componentIdx}.parts{partIdx}.defindex = cumIdx;
        model.components{componentIdx}.parts{partIdx}.defidx = cumIdx;
    end
end
    
% set up one component model
model.numcomponents = numcomponents;
model.components{1}.rootindex = 1;
model.components{1}.rootidx = 1;
model.components{1}.offsetindex = 1;
model.components{1}.offsetidx = 1;
model.components{1}.dim = 14 + sum(model.blocksizes);
model.components{1}.numblocks = 2;

% initialize the rest of the model structure
model.interval = 10;
model.numblocks = 14;
model.maxsize = [-realmax, -realmax];
for i=1:length(model.rootfilters)
    fsize = model.rootfilters{i}.size;
    if (fsize(1) > model.maxsize(1))
        model.maxsize(1) = fsize(1);
    end
    if (fsize(2) > model.maxsize(2))
        model.maxsize(2) = fsize(2);
    end
end
for i=1:length(model.partfilters)
    fsize = size(model.partfilters{i}.w);
    if (fsize(1) > model.maxsize(1))
        model.maxsize(1) = fsize(1);
    end
    if (fsize(2) > model.maxsize(2))
        model.maxsize(2) = fsize(2);
    end
end
if model.maxsize(1) == -realmax
    model.maxsize(1) = 0;
end
if model.maxsize(2) == -realmax
    model.maxsize(2) = 0;
end
model.minsize = [realmax, realmax];
for i=1:length(model.rootfilters)
    fsize = model.rootfilters{i}.size;
    if (fsize(1) < model.minsize(1))
        model.minsize(1) = fsize(1);
    end
    if (fsize(2) < model.minsize(2))
        model.minsize(2) = fsize(2);
    end
end
for i=1:length(model.partfilters)
    fsize = size(model.partfilters{i}.w);
    if (fsize(1) < model.minsize(1))
        model.minsize(1) = fsize(1);
    end
    if (fsize(2) < model.minsize(2))
        model.minsize(2) = fsize(2);
    end
end
if model.minsize(1) == realmax
    model.minsize(1) = 0;
end

if model.minsize(2) == realmax
    model.minsize(2) = 0;
end

model.padx = ceil(model.maxsize(2)/2+1);
model.pady = ceil(model.maxsize(1)/2+1);