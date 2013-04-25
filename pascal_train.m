function model = pascal_train(cls, n)

% model = pascal_train(cls)
% Train a model using the PASCAL dataset.

globals; 
[pos, allNeg] = pascal_data(cls);
cacheFeaturePyramids(pos, 8, 10);               % sbin 8, interval 10
cacheFeaturePyramids(allNeg, 8, 10);               % sbin 8, interval 10

% train root filter using warped positives & random negatives
try
  load([cachedir cls '_random']);
catch
  model = initmodel_old(pos);
  model = train_old(cls, model, pos, allNeg);
  save([cachedir cls '_random_orig'], 'model');
end

% PUT YOUR CODE HERE
% TODO: Train the rest of the DPM (latent root position, part filters, ...)

model = initmodel(pos);
model = initPartFilter(model, 6);
ITER = 2;
DATAMINE_ITER = 2;
neg = allNeg;
for iter = 1:ITER
    iter
    % latent root position
   %  pos = relabelpos(pos, model);
    
    for datamineIter = 1:DATAMINE_ITER
        datamineIter
        % get new negative data while removing easy ones
        updateNegCacheTic = tic;
        neg = updateNegCache(allNeg, neg, model, length(allNeg));
        updateNegCacheTime = toc(updateNegCacheTic)
        % train model
        
        trainTic = tic;
        model = train(cls, model, pos, neg);                       % root filter training
        trainTime = toc(trainTic)
    end
end
save([cachedir cls '_random'], 'model');