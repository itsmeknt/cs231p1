function model = train(name, model, pos, neg )

% model = train(name, model, pos, neg)
% Train LSVM. (For now it's just an SVM)
% 
  
% SVM learning parameters
C = 0.002*model.numcomponents;
J = 1;

maxsize = 2^28;
% approximate bound on the number of examples used in each iteration
dim = 0;
for i = 1:model.numcomponents
  dim = max(dim, model.components{i}.dim);
end
maxnum = floor(maxsize / (dim * 4));

globals;
hdrfile = [tmpdir name '.hdr'];
datfile = [tmpdir name '.dat'];
modfile = [tmpdir name '.mod'];
inffile = [tmpdir name '.inf'];
lobfile = [tmpdir name '.lob'];

labelsize = 5;  % [label id level x y]

% approximate bound on the number of examples used in each iteration
dim = 0;
for i = 1:model.numcomponents
  dim = max(dim, model.components{i}.dim);
end

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
num=0;
num = num+serializeFeatureToFile(pos, model, num, fid, true); 
num = num+serializeFeatureToFile(neg, model, num, fid, false); 
%num = num+poswarp(name, model, 1, pos, fid);

% Add random negatives
%num = num + negrandom(name, 1, model, 1, neg, maxnum-num, fid);
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
pixels = model.minsize * model.sbin;
minsize = prod(pixels);
num = 0;
for i=1:length(posOrNegs)
    t = tic;
    if isPos
        % do latent root positioning
        bbox = [posOrNegs(i).x1 posOrNegs(i).y1 posOrNegs(i).x2 posOrNegs(i).y2];
        % skip small examples
        if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
            continue
        end
        
        im = color(imread(posOrNegs(i).im));
        [imCropped, bboxCropped] = croppos(im, bbox);
        [featsCropped scale] = featpyramid(imCropped, model.sbin, model.interval);
        [detectionBboxes bestDetection] = detect(featsCropped, scale, model, 0, true, bboxCropped);
        if isempty(bestDetection)
            continue
        end
        c = bestDetection(1).component;
        l = bestDetection(1).level;
        paddedRfeat = padarray(featsCropped{l}, [model.pady model.padx 0], 0);
        rloc = bestDetection(1).rootLoc;
        rsize = model.rootfilters{model.components{bestDetection(1).component}.rootindex}.size;
        rootFeat = symmetrizeFeature(paddedRfeat(rloc(1):rloc(1)+rsize(1)-1, rloc(2):rloc(2)+rsize(2)-1, :), rsize);
        %c
        %size(rootFeat)
        plocs = bestDetection(1).partLocs;
        
        bboxes = zeros(size(detectionBboxes,1), 4);
        bboxes(1,:) = bboxCropped;
        bboxes = [bboxes, detectionBboxes(:, 1:4)];
        %showboxes(im, bboxes);
        paddedPfeat = padarray(featsCropped{l-model.interval}, [2*model.pady 2*model.padx 0], 0);
        
        % now do the same for flipped image
        imCroppedFlipped = imCropped(:,end:-1:1,:);
        [featsCroppedFlipped scales] = featpyramid(imCroppedFlipped, model.sbin, model.interval);
        bboxCroppedFlipped = bboxCropped;
        oldx1 = bboxCroppedFlipped(1);
        oldx2 = bboxCroppedFlipped(3);
        bboxCroppedFlipped(1) = size(imCroppedFlipped,2) - oldx2 + 1;
        bboxCroppedFlipped(3) = size(imCroppedFlipped,2) - oldx1 + 1;
        
        [detectionFlippedBboxes bestDetectionFlipped] = detect(featsCroppedFlipped, scale, model, 0, true, bboxCroppedFlipped);
        if isempty(bestDetectionFlipped)
            continue
        end
        cFlipped = bestDetectionFlipped(1).component;
        levelFlipped = bestDetectionFlipped(1).level;
        paddedRfeatFlipped = padarray(featsCroppedFlipped{levelFlipped}, [model.pady model.padx 0], 0);
        rlocFlipped = bestDetectionFlipped(1).rootLoc;
        rsizeFlipped = model.rootfilters{model.components{bestDetectionFlipped(1).component}.rootindex}.size;
        rootFeatFlipped = symmetrizeFeature(paddedRfeatFlipped(rlocFlipped(1):rlocFlipped(1)+rsizeFlipped(1)-1, rlocFlipped(2):rlocFlipped(2)+rsizeFlipped(2)-1, :), rsizeFlipped);
        
        plocsFlipped = bestDetectionFlipped(1).partLocs;
        paddedPfeatFlipped = padarray(featsCroppedFlipped{levelFlipped-model.interval}, [2*model.pady 2*model.padx 0], 0);
    else
        feat = loadFeaturePyramidCache(posOrNegs(i).id, posOrNegs(i).im, model.sbin, model.interval);
        % root filter features
        l = posOrNegs(i).level;
        rloc = posOrNegs(i).rootLoc;
        c = posOrNegs(i).component;
        plocs = posOrNegs(i).partLocs;
        rsize = model.rootfilters{model.components{c}.rootindex}.size;
        paddedRfeat = padarray(feat{l}, [model.pady model.padx 0], 0);
        rootFeat = symmetrizeFeature(paddedRfeat(rloc(1):rloc(1)+rsize(1)-1, rloc(2):rloc(2)+rsize(2)-1, :), rsize);
        
        paddedPfeat = padarray(feat{l-model.interval}, [2*model.pady 2*model.padx 0], 0);
    end
    
    
    
    % part filter/deform features
    defFeat = cell(model.numparts, 1);
    defFeatFlipped = cell(model.numparts, 1);
    partFeat = cell(model.numparts, 1);
    partFeatFlipped = cell(model.numparts, 1);
    for j = 1:model.numparts
        if (model.partfilters{model.components{c}.parts{j}.partindex}.fake)
            continue;
        end
        psize = size(model.partfilters{model.components{c}.parts{j}.partindex}.w);
        anchor = model.defs{model.components{c}.parts{j}.defindex}.anchor;
        
        partloc = plocs(j,:);
        pdef = (2*rloc + [anchor(2) anchor(1)] - [1 1]) - partloc;
        defFeat{j} = -([pdef(2)*pdef(2) pdef(2) pdef(1)*pdef(1) pdef(1)]);
        partFeat{j} = paddedPfeat(partloc(1):partloc(1)+psize(1)-1, partloc(2):partloc(2)+psize(2)-1, :);
        if (isPos)
            partlocFlipped = plocsFlipped(j,:);
            pDefFlipped = (2*rloc + [anchor(2) anchor(1)] - [1 1]) - partlocFlipped;
            defFeatFlipped{j} = -([pDefFlipped(2)*pDefFlipped(2) pDefFlipped(2) pDefFlipped(1)*pDefFlipped(1) pDefFlipped(1)]);
            partFeatFlipped{j} = paddedPfeatFlipped(partlocFlipped(1):partlocFlipped(1)+psize(1)-1, partlocFlipped(2):partlocFlipped(2)+psize(2)-1, :);
        end
        
        partnerPartIdx = model.partfilters{model.components{c}.parts{j}.partindex}.partner;
        if partnerPartIdx > 0
            partnerPsize = size(model.partfilters{partnerPartIdx}.w);
            partnerAnchor = model.defs{partnerPartIdx}.anchor;
            
            partnerLoc = plocs(model.partfilters{partnerPartIdx}.partnumber,:);
            partnerPdef = partnerLoc - (2*rloc + [partnerAnchor(2) partnerAnchor(1)] - [1 1]);
            partnerDef = -[partnerPdef(2)^2, partnerPdef(2), partnerPdef(1)^2, partnerPdef(1)];
            defFeat{j} = defFeat{j} + partnerDef;
            
            partnerFeat = paddedPfeat(partnerLoc(1):partnerLoc(1)+partnerPsize(1)-1, partnerLoc(2):partnerLoc(2)+partnerPsize(2)-1, :);
            partFeat{j} = partFeat{j} + flipfeat(partnerFeat);
            
            if (isPos)
                partnerLocFlipped = plocsFlipped(model.partfilters{partnerPartIdx}.partnumber,:);
                partnerPdefFlipped = partnerLocFlipped - (2*rloc + [partnerAnchor(2) partnerAnchor(1)] - [1 1]);
                partnerDefFlipped = -[partnerPdefFlipped(2)^2, partnerPdefFlipped(2), partnerPdefFlipped(1)^2, partnerPdefFlipped(1)];
                defFeatFlipped{j} = defFeatFlipped{j} + partnerDefFlipped;
                
                partnerFeatFlipped = paddedPfeatFlipped(partnerLocFlipped(1):partnerLocFlipped(1)+partnerPsize(1)-1, partnerLocFlipped(2):partnerLocFlipped(2)+partnerPsize(2)-1, :);
                partFeat{j} = partFeat{j} + flipfeat(partnerFeatFlipped);
            end
        else
            partFeat{j} = symmetrizeFeature(partFeat{j}, psize);
            
            if (isPos)
                partFeatFlipped{j} = symmetrizeFeature(partFeatFlipped{j}, psize);
            end
        end
    end
    
    yc = round(rloc(1) + rsize(1)/2 - model.pady);
    xc = round(rloc(2) + rsize(2)/2 - model.padx);
    
    
    if isPos
        classVal = 1;
        num = num+1;
        serializeFeature(rootFeat, partFeat, defFeat, model, c, classVal, baseFeatureId+num, l, xc, yc, fid);
        num = num+1;
        serializeFeature(rootFeatFlipped, partFeatFlipped, defFeatFlipped, model, cFlipped, classVal, baseFeatureId+num, levelFlipped, xc, yc, fid);
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

header = [classVal featureId level xc yc model.components{componentIdx}.numblocks model.components{componentIdx}.dim];
fwrite(fid, header, 'int32');
buf1 = [oblocklabel; 1; rblocklabel; rootFeat(:)];
fwrite(fid, buf1, 'single');

for i=1:model.numparts
    if (model.partfilters{model.components{componentIdx}.parts{i}.partindex}.fake)
        continue;
    end
    pidx = model.components{componentIdx}.parts{i}.partindex;
    pblocklabel = model.partfilters{pidx}.blocklabel;
    didx = model.components{componentIdx}.parts{i}.defindex;
    dblocklabel = model.defs{didx}.blocklabel;
    buf = [pblocklabel; partFeats{i}(:); dblocklabel; deformFeats{i}'];
    fwrite(fid, buf, 'single');
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
function num = negrandom(name, t, model, c, neg, maxnum, fid)
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
  fprintf('%s: iter %d: random negatives: %d/%d\n', name, t, i, numneg);
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