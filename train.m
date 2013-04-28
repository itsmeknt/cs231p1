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
save([cachedir name '_model2'], 'model');
end

function num = serializeFeatureToFile(posOrNegs, model, baseFeatureId, fid, isPos)
if (isPos)
    [bboxes bestDetections ambiguousCases] = detectParallel(posOrNegs, model, 0, true, true);
    for i=1:length(posOrNegs)  
        bestDetection = bestDetections{i};
        posOrNegs(i).bestRootLoc = bestDetection(1).rootLoc;
        posOrNegs(i).bestPartLoc = bestDetection(1).partLocs;
        posOrNegs(i).bestRootLevel = bestDetection(1).level;
        posOrNegs(i).bestComponentIdx = bestDetection(1).component;
    end
end
num = 0;
for i=1:length(posOrNegs)
    t = tic;
    feat = loadFeaturePyramidCache(posOrNegs(i).id, posOrNegs(i).im, model.sbin, model.interval);

    
    % root filter features
    l = posOrNegs(i).bestRootLevel;
    rloc = posOrNegs(i).bestRootLoc;
    c = posOrNegs(i).bestComponentIdx;
    rsize = model.rootfilters{model.components{c}.rootindex}.size;
    
    posOrNegs(i).id
    l
    size(feat)
    rloc
    rsize
    model.pady
    model.padx
    feat
    size(feat{l})
    paddedRfeat = padarray(feat{l}, [model.pady model.padx 0], 0);
    rootFeat = symmetrizeFeature(paddedRfeat(rloc(1):rloc(1)+rsize(1)-1, rloc(2):rloc(2)+rsize(2)-1, :), rsize);
    yc = round(rloc(1) + rsize(1)/2);
    xc = round(rloc(2) + rsize(2)/2);
    % part filter features
    plocs = posOrNegs(i).bestPartLoc;
    defFeat = cell(model.numparts, 1);
    partFeat = cell(model.numparts, 1);
    partFeatFlipped = cell(model.numparts, 1);
    paddedPfeat = padarray(feat{l-model.interval}, [2*model.pady 2*model.padx 0], 0);
    for j = 1:model.numparts
        anchor = model.defs{model.components{c}.parts{j}.defidx}.anchor;
        pDef = plocs(j,:) - (2*rloc + anchor);
        defFeat{j} = -abs([pDef(2) pDef(1) pDef(2)*pDef(2) pDef(1)*pDef(1)]);
        
        psize = size(model.partfilters{model.components{c}.parts{j}.partidx}.w);
        partloc = plocs(j,:);
        j
        partloc
        psize
        size(paddedPfeat)
        partFeat{j} = paddedPfeat(partloc(1):partloc(1)+psize(1)-1, partloc(2):partloc(2)+psize(2)-1, :);
        partnerPartIdx = model.partfilters{model.components{c}.parts{j}.partindex}.partner;
        if partnerPartIdx > 0
            partnerLoc = plocs(model.partfilters{partnerPartIdx}.partnumber,:);
            partnerAnchor = model.defs{partnerPartIdx}.anchor;
            partnerAnchorAbsolute = 2*([model.pady model.padx] + rsize)  + partnerAnchor;
            partnerDef = -abs([partnerLoc(2)-partnerAnchorAbsolute(2), partnerLoc(1)-partnerAnchorAbsolute(1), (partnerLoc(2)-partnerAnchorAbsolute(2))^2, (partnerLoc(1)-partnerAnchorAbsolute(2))^2]);
            defFeat{j} = defFeat{j} + partnerDef(end:-1:1);
            
            partnerFeat = paddedPfeat(partnerLoc(1):partnerLoc(1)+psize(1)-1, partnerLoc(2):partnerLoc(2)+psize(2)-1, :);
            partFeat{j} = partFeat{j} + flipfeat(partnerFeat);
        else
            partFeat{j} = symmetrizeFeature(partFeat{j});
        end
        
        partFeatFlipped{j} = flipfeat(partFeat{j});
    end
    
    if isPos
        classVal = 1;
        num = num+1;
        serializeFeature(rootFeat, partFeat, defFeat, model, c, classVal, baseFeatureId+num, l, xc, yc, fid);
        num = num+1;
        serializeFeature(flipfeat(rootFeat), partFeatFlipped, defFeat, model, c, classVal, baseFeatureId+num, l, xc, yc, fid);
    else
        classVal = -1;
        num = num+1;
        serializeFeature(rootFeat, partFeat, defFeat, model, c, classVal, baseFeatureId+num, l, xc, yc, fid);
    end
    
    serializeTime = toc(t);
    [i length(posOrNegs) serializeTime]
end
end

function feat = symmetrizeFeature(feat, filterSize)
width1 = ceil(filterSize(2)/2);
width2 = floor(filterSize(2)/2);
feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
feat = feat(:,1:width1,:);
end

function fid = serializeFeature(rootFeat, partFeats, deformFeats, model, componentIdx, classVal, featureId, level, xc, yc, fid)
ridx = model.components{componentIdx}.rootindex;
oidx = model.components{componentIdx}.offsetindex;
rblocklabel = model.rootfilters{ridx}.blocklabel;
oblocklabel = model.offsets{oidx}.blocklabel;

header = [classVal; featureId; level; xc; yc; model.components{componentIdx}.numblocks; model.components{componentIdx}.dim];
fwrite(fid, header, 'int32');
buf1 = [oblocklabel; 1; rblocklabel; rootFeat(:)];
fwrite(fid, buf1, 'single');
for i=1:model.numparts
    pidx = model.components{componentIdx}.parts{i}.partidx;
    if (model.partfilters{pidx}.fake)
        continue;
    end
    pblocklabel = model.partfilters{pidx}.blocklabel;
    didx = model.components{componentIdx}.parts{i}.defidx;
    dblocklabel = model.defs{didx}.blocklabel;
    buf = [pblocklabel; model.partfilters{pidx}.w(:); dblocklabel; model.defs{didx}.w(:)];
    fwrite(fid, buf, 'single');
end
end