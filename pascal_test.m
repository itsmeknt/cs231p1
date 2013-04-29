function [boxes] = pascal_test(cls, model, testset, suffix)

% [boxes] = pascal_test(cls, model, testset, suffix)
% Compute bounding boxes in a test set.
% boxes are bounding boxes from root placements

globals;
pascal_init;
ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

% run detector in each image
try
  load([cachedir cls '_boxes_' testset '_' suffix]);
catch
  for i = 1:length(ids);
    fprintf('%s: testing: %s %s, %d/%d\n', cls, testset, VOCyear, ...
            i, length(ids));
    [feat, scales] = loadFeaturePyramidCache(ids{i}, sprintf(VOCopts.imgpath, ids{i}), model.sbin, model.interval);
    b = detect(feat, scales, model, model.thresh, false, false);
    if ~isempty(b)
      b1 = b(:,[1 2 3 4 end]);
      b1 = clipboxes(zeros(model.sbin*size(feat{1+model.interval},1), model.sbin*size(feat{1+model.interval},2)), b1);
      boxes{i} = nms(b1, 0.5);
    else
      boxes{i} = [];
    end
    %showboxes(im, boxes{i});
  end
  save([cachedir cls '_boxes_' testset '_' suffix], 'boxes');
end
