function model=initParts(model,component, numParts)

% This function initializes the part filters, given a trained root filter.
% The function enforces certain characteristics in the part. These are
% clearly mentioned in comments in the file.
model.numparts = numParts;
rootSize=model.rootfilters{component}.size;
rootWeights=imresize(model.rootfilters{component}.w,2,'bicubic');

% We wish to compute the energy using positive weights only.

PosWeights=max(rootWeights,0);
EnergyMat=sum(PosWeights.^2,3); % l2 norm of each 31 dimensional weight

% Make the Energy Matrix symmetric by flipping and adding to itself.

EnergyMat=EnergyMat+EnergyMat(:,end:-1:1);

% Each part should satisfy conform to bounds with respect to height, width,
% aspect ratio and fraction of root filter area covered.

rootArea=rootSize(1)*rootSize(2)*4;

numSizes=0;
for h=3:1:size(rootWeights,1)-2
    for w=3:1:size(rootWeights,2)-2
        if (w*h>=1*rootArea/model.numparts && w*h<=1.2*rootArea/model.numparts && abs(h-w)<=4)   % constraints
            numSizes=numSizes+1;
            validSizes{numSizes}=fspecial('average', [h w]); % normalized averaging function
        end
    end
end


% Continue to add parts till model.numparts parts have been added

numAdded=0;
mid=size(rootWeights,2)/2;
while numAdded<model.numparts
    
    for i=1:numSizes
        % Compute the normalized energy score of each valid
        % part filter size at every location of the root
        convScore=conv2(EnergyMat,validSizes{i},'valid');
        copyScore=convScore;
        
        % Zero out scores for placement of parts that would lead to
        % overlapping symmetric parts
        Index=max(mid-size(validSizes{i},2)+2,1);
        convScore(:,Index:end)=-inf;
        
        % Set all positions as temporarily invalid, if only one filter
        % remains to be placed. Later we will allow for centered placements
        % in addition to what is allowed at the end of this step. Therefore
        % at most one centered part can be placed.
        if (numAdded==model.numparts-1)
            convScore(:,:)=-Inf;
        end
        
        % Set scores for centered placement of parts to be the original
        % scores. Parts can only be centered if they have even width.
         m = size(validSizes{i}, 2)/2;
         if m == round(m)
            convScore(:,mid-(size(validSizes{i},2)/2)+1)=copyScore(:,mid-(size(validSizes{i},2)/2)+1);
         end
        
        
        % Now find the maximum score for the shape
        
        [dummy, y_inds]=max(convScore);
        [maxSc, x]=max(dummy);
        y=y_inds(x);
        ys(i) = y;
        xs(i) = x;
        vs(i) = maxSc;
    end
    
    [dummy i] = max(vs);
    x = xs(i);
    y = ys(i);
    % Check if the placement x is centered
    sym_x=size(rootWeights,2)+2-size(validSizes{i},2)-x; % x for the symmetrically placed part
    if (x==sym_x)
        centered=true;
        width = ceil(size(validSizes{i},2)/2);
    else
        centered=false;
        width = ceil(size(validSizes{i},2));
    end
    
    
    % now add the parts (1 addition if centered, 2 otherwise). Each
    % time a part is added the corresponding area in the energy matrix
    % is zeroed out
    numAdded=numAdded+1;
    [model partner1] = updateModel(model, numAdded, rootWeights, x, y, validSizes, i, width, component, false);
    EnergyMat(y:y+size(validSizes{i},1)-1,x:x+size(validSizes{i},2)-1)=0;
    
    if ~centered
        numAdded=numAdded+1;
        [model partner2] = updateModel(model, numAdded, rootWeights, sym_x, y, validSizes, i, width, component, true);
        EnergyMat(y:y+size(validSizes{i},1)-1,sym_x:sym_x+size(validSizes{i},2)-1)=0;
        
        model.partfilters{partner1}.partner = partner2;
        model.partfilters{partner2}.partner = partner1;
    end
end
end


function [model partnerIdx] = updateModel(model, partIdx, rootWeights, x, y, validSizes, i, width, componentIdx, fake)
cumIdx = length(model.partfilters) + 1;
model.partfilters{cumIdx}.w=rootWeights(y:y+size(validSizes{i},1)-1,x:x+size(validSizes{i},2)-1,:);
model.partfilters{cumIdx}.fake = fake;
model.partfilters{cumIdx}.partner = 0;

model.defs{cumIdx}.anchor=[x y];
model.defs{cumIdx}.w=[0.1 0 0.1 0];                   % initially only allow quadratic terms

model.components{componentIdx}.parts{partIdx}.partindex = cumIdx;
model.components{componentIdx}.parts{partIdx}.partidx = cumIdx;
model.components{componentIdx}.parts{partIdx}.defindex = cumIdx;
model.components{componentIdx}.parts{partIdx}.defidx = cumIdx;

model.partfilters{cumIdx}.partnumber = partIdx;
    
if ~fake
    model.partfilters{cumIdx}.blocklabel = length(model.blocksizes)+1;
    model.blocksizes(model.partfilters{cumIdx}.blocklabel) = width*size(validSizes{i},1)*31;
    model.regmult(model.partfilters{cumIdx}.blocklabel) = 1;
    model.learnmult(model.partfilters{cumIdx}.blocklabel) = 1;
    model.lowerbounds{model.partfilters{cumIdx}.blocklabel} = -100*ones(model.blocksizes(model.partfilters{cumIdx}.blocklabel),1);
    model.numblocks = model.numblocks+1;
    
    model.defs{cumIdx}.blocklabel = length(model.blocksizes)+1;
    model.blocksizes(model.defs{cumIdx}.blocklabel) = numel(model.defs{cumIdx}.w);
    model.regmult(model.defs{cumIdx}.blocklabel) = 10;
    model.learnmult(model.defs{cumIdx}.blocklabel) = 0.1;
    model.lowerbounds{model.defs{cumIdx}.blocklabel} = [0.1, -100, 0.1, -100];
    model.numblocks = model.numblocks+1;
    
    model.components{componentIdx}.dim = model.numblocks + sum(model.blocksizes);
end
partnerIdx = cumIdx;
end
        
        
        
        
        
    
   
