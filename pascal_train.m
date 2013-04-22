function model = pascal_train(cls, n)

% model = pascal_train(cls)
% Train a model using the PASCAL dataset.

globals; 
[pos, neg] = pascal_data(cls);

% train root filter using warped positives & random negatives
try
  load([cachedir cls '_random']);
catch
  model = initmodel(pos);
  model = train(cls, model, pos, neg);
  save([cachedir cls '_random_orig'], 'model');
end

% PUT YOUR CODE HERE
% TODO: Train the rest of the DPM (latent root position, part filters, ...)

ITER = 5;
DATAMINE_ITER = 5;
allNeg = neg;
size = length(neg);
for iter = 1:ITER
    % latent root position
    pos = relabelpos(pos, model);
    
    for datamineIter = 1:DATAMINE_ITER
        % get new negative data while removing easy ones
        neg = updateNegCache(allNeg, neg, model, size);
        
        % train model
        model = initmodel(pos);
        model = train(cls, model, nos, neg);                       % root filter training
        model = initPartFilter(model, 6);                             %
                                                                      % compute part pos
        model = trainPartFilters(model, pos, neg);                 % part filter + deformation cost traiing
    end
end
save([cachedir cls '_random'], 'model');