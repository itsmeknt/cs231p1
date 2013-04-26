function model = train(name, model, pos, neg )

% model = train(name, model, pos, neg)
% Train LSVM. (For now it's just an SVM)
% 
  
% SVM learning parameters
C = 0.002*model.numcomponents;
J = 1;

maxsize = 2^28;
globals;
hdrfile = [tmpdir name '2.hdr'];
datfile = [tmpdir name '2.dat'];
modfile = [tmpdir name '2.mod'];
inffile = [tmpdir name '2.inf'];
lobfile = [tmpdir name '2.lob'];

labelsize = 5;  % [label id level x y]
negpos = 0;     % last position in data mining

% approximate bound on the number of examples used in each iteration
dim = 0;
for i = 1:model.numcomponents
  dim = max(dim, model.components{i}.dim);
end
maxnum = floor(maxsize / (dim * 4));

% Reset some of the tempoaray files, just in case
% reset data file
fid = fopen(datfile, 'wb');
fclose(fid);
% reset header file
writeheader(hdrfile, 0, labelsize, model);  
% reset info file
fid = fopen(inffile, 'w');
fclose(fid);
% reset initial model 
fid = fopen(modfile, 'wb');
fwrite(fid, zeros(sum(model.blocksizes), 1), 'double');
fclose(fid);
% reset lower bounds
writelob(lobfile, model)


% Find the positive examples and safe them in the data file
fid = fopen(datfile, 'w');
num = serializeFeatureToFile(pos, model, 0, fid, true); 
% num = poswarp(name, model, 1, pos, fid);

% Add random negatives
num = num+serializeFeatureToFile(neg, model, length(pos), fid, false); 
% num = num + negrandom(name, model, 1, neg, maxnum-num, fid);
fclose(fid);
        
% learn model
writeheader(hdrfile, num, labelsize, model);
% reset initial model 
fid = fopen(modfile, 'wb');
fwrite(fid, zeros(sum(model.blocksizes), 1), 'double');
fclose(fid);

% Call the SVM learning code
cmd = sprintf('./learn %.4f %.4f %s %s %s %s %s', ...
              C, J, hdrfile, datfile, modfile, inffile, lobfile);
fprintf('executing: %s\n', cmd);
status = unix(cmd);
if status ~= 0
  fprintf('command `%s` failed\n', cmd);
  keyboard;
end
    
fprintf('parsing model\n');
blocks = readmodel(modfile, model);
model = parsemodel(model, blocks);
[labels, vals, unique] = readinfo(inffile);
    
% compute threshold for high recall
P = find((labels == 1) .* unique);
pos_vals = sort(vals(P));
model.thresh = pos_vals(ceil(length(pos_vals)*0.05));

% cache model
save([cachedir name '_model'], 'model');
end

function num = serializeFeatureToFile(posOrNeg, model, baseFeatureId, fid, isPos)
if (isPos)
    [dummys ambiguouss bestRootLocs bestPartLocs bestRootLevels bestComponentIdxs] = detectParallel(posOrNeg, model, 1);
    for i=1:length(posOrNeg)        
        posOrNeg(i).bestRootLoc = bestRootLocs{i};
        posOrNeg(i).bestPartLoc = bestPartLocs{i};
        posOrNeg(i).bestRootLevel = bestRootLevels{i};
        posOrNeg(i).bestComponentIdx = bestComponentIdxs{i};
    end
end
num = 0;
for i=1:length(posOrNeg)
    t = tic;
    feat = loadFeaturePyramidCache(posOrNeg(i).id);

    
    % root filter features
    l = posOrNeg(i).bestRootLevel;
    rloc = posOrNeg(i).bestRootLoc;
    c = posOrNeg(i).bestComponentIdx;
    rsize = model.rootfilters{model.components{c}.rootindex}.size;
    rootFeat = symmetrizeFeature(feat{l}(rloc(1):rloc(1)+rsize(1)-1, rloc(2):rloc(2)+rsize(2)-1, :), rsize);
    
    % part filter features
    ploc = posOrNeg(i).bestPartLoc;
    defFeat = cell(model.numparts, 1);
    partFeat = cell(model.numparts, 1);
    partFeatFlipped = cell(model.numparts, 1);
    for j = 1:model.numparts
        anchor = model.defs{model.components{c}.parts{j}.defidx}.anchor;
        pDef = ploc(j,:) - rsize - anchor;
        defFeat{j} = [pDef(2) pDef(1) pDef(2)*pDef(2) pDef(1)*pDef(1)];
        
        psize = size(model.partfilters{model.components{c}.parts{j}.partidx}.w);
        partloc = pDef + 2*rloc + anchor;
        partFeat{j} = symmetrizeFeature(feat{l-model.interval}(partloc(1):partloc(1)+psize(1)-1, partloc(2):partloc(2)+psize(2)-1, :), psize);
        partFeatFlipped{j} = flipfeat(partFeat{j});
    end
    
    if isPos
        classVal = 1;
        serializeFeature(rootFeat, partFeat, defFeat, model, c, classVal, baseFeatureId+2*i-1, fid);
        serializeFeature(flipfeat(rootFeat), partFeatFlipped, defFeat, model, c, classVal, baseFeatureId+2*i, fid);
    else
        classVal = -1;
        serializeFeature(rootFeat, partFeat, defFeat, model, c, classVal, baseFeatureId+i, fid);
    end
    
    num = num+1;
    serializeTime = toc(t)
end
end

function feat = symmetrizeFeature(feat, filterSize)
width1 = ceil(filterSize(2)/2);
width2 = floor(filterSize(2)/2);
feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
feat = feat(:,1:width1,:);
end

function fid = serializeFeature(rootFeat, partFeats, deformFeats, model, componentIdx, classVal, featureId, fid)
ridx = model.components{componentIdx}.rootindex;
oidx = model.components{componentIdx}.offsetindex;
rblocklabel = model.rootfilters{ridx}.blocklabel;
oblocklabel = model.offsets{oidx}.blocklabel;

fwrite(fid, [classVal featureId 0 0 0 2 model.components{componentIdx}.dim], 'int32');
fwrite(fid, oblocklabel, 'single');
fwrite(fid, 1, 'single');
fwrite(fid, rblocklabel, 'single');
fwrite(fid, rootFeat, 'single');
for i=1:model.numparts
    pidx = model.components{componentIdx}.parts{i}.partidx;
    pblocklabel = model.partfilters{pidx}.blocklabel;
    fwrite(fid, pblocklabel, 'single');
    fwrite(fid, partFeats{i}, 'single');
    didx = model.components{componentIdx}.parts{i}.defidx;
    dblocklabel = model.defs{didx}.blocklabel;
    fwrite(fid, dblocklabel, 'single');
    fwrite(fid, deformFeats{i}, 'single');
end
end

% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function num = poswarp(name, model, c, pos, fid)
numpos = length(pos);
warped = warppos(name, model, c, pos);
ridx = model.components{c}.rootindex;
oidx = model.components{c}.offsetindex;
rblocklabel = model.rootfilters{ridx}.blocklabel;
oblocklabel = model.offsets{oidx}.blocklabel;
dim = model.components{c}.dim;
width1 = ceil(model.rootfilters{ridx}.size(2)/2);
width2 = floor(model.rootfilters{ridx}.size(2)/2);
pixels = model.rootfilters{ridx}.size * model.sbin;
minsize = prod(pixels);
num = 0;
for i = 1:numpos
    if mod(i,100)==0
        fprintf('%s: warped positive: %d/%d\n', name, i, numpos);
    end
    bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue
    end    
    % get example
    im = warped{i};
    feat = features(im, model.sbin);
    feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
    feat = feat(:,1:width1,:);
    fwrite(fid, [1 2*i-1 0 0 0 2 dim], 'int32');
    fwrite(fid, [oblocklabel 1], 'single');
    fwrite(fid, rblocklabel, 'single');
    fwrite(fid, feat, 'single');    
    % get flipped example
    feat = features(im(:,end:-1:1,:), model.sbin);    
    feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
    feat = feat(:,1:width1,:);
    fwrite(fid, [1 2*i 0 0 0 2 dim], 'int32');
    fwrite(fid, [oblocklabel 1], 'single');
    fwrite(fid, rblocklabel, 'single');
    fwrite(fid, feat, 'single');
    num = num+2;    
end
end

% get random negative examples
function num = negrandom(name, model, c, neg, maxnum, fid)
numneg = length(neg);
rndneg = floor(maxnum/numneg);
ridx = model.components{c}.rootindex;
oidx = model.components{c}.offsetindex;
rblocklabel = model.rootfilters{ridx}.blocklabel;
oblocklabel = model.offsets{oidx}.blocklabel;
rsize = model.rootfilters{ridx}.size;
width1 = ceil(rsize(2)/2);
width2 = floor(rsize(2)/2);
dim = model.components{c}.dim;
num = 0;
for i = 1:numneg
  if mod(i,100)==0
    fprintf('%s: random negatives: %d/%d\n', name, i, numneg);
  end
  im = color(imread(neg(i).im));
  feat = features(double(im), model.sbin);  
  if size(feat,2) > rsize(2) && size(feat,1) > rsize(1)
    for j = 1:rndneg
      x = random('unid', size(feat,2)-rsize(2)+1);
      y = random('unid', size(feat,1)-rsize(1)+1);
      f = feat(y:y+rsize(1)-1, x:x+rsize(2)-1,:);
      f(:,1:width2,:) = f(:,1:width2,:) + flipfeat(f(:,width1+1:end,:));
      f = f(:,1:width1,:);
      fwrite(fid, [-1 (i-1)*rndneg+j 0 0 0 2 dim], 'int32');
      fwrite(fid, [oblocklabel 1], 'single');
      fwrite(fid, rblocklabel, 'single');
      fwrite(fid, f, 'single');
    end
    num = num+rndneg;
  end
end
end