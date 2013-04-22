function [pos, neg] = pascal_data(cls)

% [pos, neg] = pascal_data(cls)
% Get training data from the PASCAL dataset.
%
% input:
% cls - class name
%
% output:
% pos = an array of data structs of all positive examples in train and val
% neg = an array of data structs of all negative examples in train
%
% data is a struct with the following member variables:
%   .im - image file path
%   .id - data instance id
%   .x1 - optional, if there is a bounding box
%   .x2 - optional, if there is a bounding box
%   .y1 - optional, if there is a bounding box
%   .y2 - optional, if there is a bounding box

globals; 
pascal_init;

try
  load([cachedir cls '_train']);
catch
  % positive examples from train+val
  ids = textread(sprintf(VOCopts.imgsetpath, 'trainval'), '%s');
  pos = [];
  numpos = 0;
  for i = 1:length(ids);
    if mod(i,100)==0
        fprintf('%s: parsing positives: %d/%d\n', cls, i, length(ids));
    end
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    % skip difficult examples
    diff = [rec.objects(clsinds).difficult];
    clsinds(diff) = [];
    for j = clsinds(:)'
      numpos = numpos+1;
      pos(numpos).im = [VOCopts.datadir rec.imgname];
      bbox = rec.objects(j).bbox;
      pos(numpos).x1 = bbox(1);
      pos(numpos).y1 = bbox(2);
      pos(numpos).x2 = bbox(3);
      pos(numpos).y2 = bbox(4);
      pos(numpos).id = ids(i);
    end
  end

  % negative examples from train (this seems enough!)
  ids = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');
  neg = [];
  numneg = 0;
  for i = 1:length(ids);
    if mod(i,100)==0
        fprintf('%s: parsing negatives: %d/%d\n', cls, i, length(ids));
    end
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    if length(clsinds) == 0
      numneg = numneg+1;
      neg(numneg).im = [VOCopts.datadir rec.imgname];
      neg(numneg).id = ids(i);
    end
  end
  
  save([cachedir cls '_train'], 'pos', 'neg');
end  
