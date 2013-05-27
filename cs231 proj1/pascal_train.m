function model = pascal_train(cls, n)

% model = pascal_train(cls)
% Train a model using the PASCAL dataset.

'loading data...'
globals; 
[pos, neg] = pascal_data(cls);
allNeg = neg;

cacheFeaturePyramids(pos, 8, 5);               % sbin 8, interval 10
cacheFeaturePyramids(neg, 8, 5);               % sbin 8, interval 10

spos = split(pos, n);
models = cell(1,n);
% train root filter using warped positives & random negatives
'training root filter...'
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
'training on hard negative and latent root positions...'
try 
  load([cachedir cls '_hard']);
catch
  model = mergemodels(models);
  neg = updateNegCache(allNeg, neg, model, 4*length(pos));
  model = train(cls, model, pos, neg);
  save([cachedir cls '_hard'], 'model');
end


'training with parts'
% add parts and update models using latent detections & hard negatives.
try 
  load([cachedir cls '_parts']);
catch
    for i=1:n
        model = initParts(model, i, 6);
    end
    neg = updateNegCache(allNeg, neg, model, 4*length(pos));
    model = train(cls, model, pos, neg);
    save([cachedir cls '_parts', 'model');
end

'training multi iter'
% ONLY FOR MULTI-ITERATION
TRAINING_ITER = 1;
try 
  load([cachedir cls '_parts2']); % change this to
catch
    for training_iter = 1:TRAINING_ITER
        neg = updateNegCache(allNeg, neg, model, 4*length(pos));
        model = train(cls, model, pos, neg);
        save([cachedir cls '_parts2'], 'model');
    end
end

