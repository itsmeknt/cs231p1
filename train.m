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
% num = poswarp(name, model, 1, pos, fid);

% Add random negatives
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
        plocs = bestDetection(1).partLocs;
        showboxes(im, [detectionBboxes; bboxCropped, 1, 1]);
        paddedPfeat = padarray(featsCropped{l-model.interval}, [2*model.pady 2*model.padx 0], 0);
        
        % now do the same for flipped image
        imCroppedFlipped = im(:,end:-1:1,:);
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
        levelFlipped = bestDetectionFlipped(1).level;
        paddedRfeatFlipped = padarray(featsCroppedFlipped{levelFlipped}, [model.pady model.padx 0], 0);
        rlocFlipped = bestDetectionFlipped(1).rootLoc;
        rsizeFlipped = model.rootfilters{model.components{bestDetectionFlipped(1).component}.rootindex}.size;
        rootFeatFlipped = symmetrizeFeature(paddedRfeatFlipped(rlocFlipped(1):rlocFlipped(1)+rsizeFlipped(1)-1, rlocFlipped(2):rlocFlipped(2)+rsizeFlipped(2)-1, :), rsizeFlipped);
        plocsFlipped = bestDetectionFlipped(1).partLocs;
        showboxes(im, [detectionFlippedBboxes; bboxCroppedFlipped, 1, 1]);
        paddedPfeatFlipped = padarray(featsCroppedFlipped{l-model.interval}, [2*model.pady 2*model.padx 0], 0);
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
    
    
    
    % part filter features
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
        
        pDef = (2*rloc + [anchor(2) anchor(1)]) - plocs(j,:);
        defFeat{j} = -([pDef(1)*pDef(1) pDef(1) pDef(2)*pDef(2) pDef(2)]);
        partloc = plocs(j,:);
        partFeat{j} = paddedPfeat(partloc(1):partloc(1)+psize(1)-1, partloc(2):partloc(2)+psize(2)-1, :);
        if (isPos)
            pDefFlipped = (2*rloc + [anchor(2) anchor(1)]) - plocsFlipped(j,:);
            defFeatFlipped{j} = -([pDefFlipped(1)*pDefFlipped(1) pDefFlipped(1) pDefFlipped(2)*pDefFlipped(2) pDefFlipped(2)]);
            partlocFlipped = plocsFlipped(j,:);
            partFeatFlipped{j} = paddedPfeatFlipped(partlocFlipped(1):partlocFlipped(1)+psize(1)-1, partlocFlipped(2):partlocFlipped(2)+psize(2)-1, :);
        end
        
        partnerPartIdx = model.partfilters{model.components{c}.parts{j}.partindex}.partner;
        if partnerPartIdx > 0
            partnerAnchor = model.defs{partnerPartIdx}.anchor;
            partnerAnchorAbsolute = 2*([model.padx model.pady] + [rsize(2) rsize(1)])  + partnerAnchor;
            
            partnerLoc = plocs(model.partfilters{partnerPartIdx}.partnumber,:);
            partnerDef = -[(partnerLoc(2)-partnerAnchorAbsolute(2))^2, partnerLoc(2)-partnerAnchorAbsolute(2), (partnerLoc(1)-partnerAnchorAbsolute(2))^2, partnerLoc(1)-partnerAnchorAbsolute(1)];
            defFeat{j} = defFeat{j} + partnerDef;
            
            partnerFeat = paddedPfeat(partnerLoc(1):partnerLoc(1)+psize(1)-1, partnerLoc(2):partnerLoc(2)+psize(2)-1, :);
            partFeat{j} = partFeat{j} + flipfeat(partnerFeat);
            
            if (isPos)
                partnerLocFlipped = plocsFlipped(model.partfilters{partnerPartIdx}.partnumber,:);
                partnerDefFlipped = -[(partnerLocFlipped(2)-partnerAnchorAbsolute(2))^2, partnerLocFlipped(2)-partnerAnchorAbsolute(2), (partnerLocFlipped(1)-partnerAnchorAbsolute(2))^2, partnerLocFlipped(1)-partnerAnchorAbsolute(1)];
                defFeatFlipped{j} = defFeatFlipped{j} + partnerDefFlipped;
                
                partnerFeatFlipped = paddedPfeatFlipped(partnerLocFlipped(1):partnerLocFlipped(1)+psize(1)-1, partnerLocFlipped(2):partnerLocFlipped(2)+psize(2)-1, :);
                partFeat{j} = partFeat{j} + flipfeat(partnerFeatFlipped);
            end
        else
            partFeat{j} = symmetrizeFeature(partFeat{j}, psize);
            
            if (isPos)
                partFeatFlipped{j} = symmetrizeFeature(partFeatFlipped{j}, psize);
            end
        end
    end
    
    yc = round(rloc(1) + rsize(1)/2);
    xc = round(rloc(2) + rsize(2)/2);
    
    % save best feature
    classVal = 1;
    num = num+1;
    serializeFeature(rootFeat, partFeat, defFeat, model, c, classVal, baseFeatureId+num, l, xc, yc, fid);
    % if its positive latent, also save the flipped feature
    if isPos
        num = num+1;
        serializeFeature(rootFeatFlipped, partFeatFlipped, defFeatFlipped, model, c, classVal, baseFeatureId+num, l, xc, yc, fid);
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
    if (model.partfilters{model.components{componentIdx}.parts{i}.partindex}.fake)
        continue;
    end
    pidx = model.components{componentIdx}.parts{i}.partidx;
    pblocklabel = model.partfilters{pidx}.blocklabel;
    didx = model.components{componentIdx}.parts{i}.defidx;
    dblocklabel = model.defs{didx}.blocklabel;
    buf = [pblocklabel; model.partfilters{pidx}.w(:); dblocklabel; model.defs{didx}.w(:)];
    fwrite(fid, buf, 'single');
end
end