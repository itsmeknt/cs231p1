function model = pascal_train(cls, n)

% model = pascal_train(cls)
% Train a model using the PASCAL dataset.

globals; 
[pos, neg] = pascal_data(cls);
allNeg = neg;

cacheFeaturePyramids(pos, 8, 10);               % sbin 8, interval 10
cacheFeaturePyramids(neg, 8, 10);               % sbin 8, interval 10

spos = split(pos, n);
models = cell(1,n);
% train root filter using warped positives & random negatives
try
  load([cachedir cls '_random']);
catch
  for i=1:n
    models{i} = initmodel(spos{i});
    models{i} = train_old(cls, models{i}, spos{i}, neg);
  end
  save([cachedir cls '_random'], 'models');
end

% PUT YOUR CODE HERE
% TODO: Train the rest of the DPM (latent root position, part filters, ...)


% merge models and train using latent detections & hard negatives
try 
  load([cachedir cls '_hard']);
catch
  model = mergemodels(models);
  neg = updateNegCache(allNeg, neg, model, length(neg));
  model = train_old(cls, model, pos, neg(1:200));
  save([cachedir cls '_hard'], 'model');
end


% add parts and update models using latent detections & hard negatives.
try 
  load([cachedir cls '_parts']);
catch
  for i=1:n
    model = initParts(model, i, 6);
  end 
  neg = updateNegCache(allNeg, neg, model, length(neg));
  model = train(cls, model, pos, neg(1:200)); 
  save([cachedir cls '_parts'], 'model');
end
