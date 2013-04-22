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

[fPyramids uids scale] = getPyramidFeats(model, cls);
fPos = pos;
fNeg = neg;
for iter = 1:ITER
    % latent root position
    fPos = relabelpos(fPos, model);
    
    for datamineIter = 1:DATAMINE_ITER
        % get new negative data
        
        
        % part initialization
        model = initmodel(fPos);
        model = train(cls, model, fPos, neg);                       % root filter training
        model = initPartFilter(model, 6);                             %
                                                                      % compute part pos
        model = trainPartFilters(model, fPos, neg);                 % part filter + deformation cost traiing
        
        % remove negatives
    end
end
save([cachedir cls '_random'], 'model');