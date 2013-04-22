function [pyramidFeats ids scale] = getPyramidFeats(model, cls)

% [pos, neg] = pascal_data(cls)
% Get training data from the PASCAL dataset.

globals; 
pascal_init;

filename = '_pyramidFeats';

try
  load([cachedir cls filename]);
catch
  % positive examples from train+val
  ids = textread(sprintf(VOCopts.imgsetpath, 'trainval'), '%s');
  pyramidFeats = cell(length(ids), 1);
  for i = 1:length(ids);
    if mod(i,100)==0
        fprintf('%s: featurizing instances: %d/%d\n', cls, i, length(ids));
    end
    im = imread(sprintf(VOCopts.imgpath, ids{i})); 
    [feat scale] = featpyramid(im, model.sbin, model.interval);
    pyramidFeats{i} = feat;
  end
  
  save([cachedir cls filename], 'pyramidFeats');
end  
