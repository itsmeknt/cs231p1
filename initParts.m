function model=initParts(model,component)

% This function initializes the part filters, given a trained root filter.
% The function enforces certain characteristics in the part. These are
% clearly mentioned in comments in the file.

rootSize=model.rootfilters{component}.size;
rootWeights=imresize(model.rootfilters{component}.w,2,'bicubic');

% We wish to compute the energy using positive weights only.

PosWeights=max(rootWeights,0);
EnergyMat=sum(PosWeights.^2,3); % l2 norm of each 31 dimensional weight

% Make the Energy Matrix symmetric by flipping and adding to itself.

EnergyMat=EnergyMat+EnergyMat(:,end:-1:1);

% Each part should satisfy conform to bounds with respect to height, width,
% aspect ratio and fraction of root filter area covered.

rootArea=rootSize(1)*rootSize(2);

numSizes=0;
for h=3:1:rootSize(1)-2
    for w=3:1:rootSize(2)-2
        if (w*h>=0.8*rootArea/model.numparts && w*h<=1.2*rootArea/model.numparts && abs(h-w)<=4)   % constraints
            numSizes=numSizes+1;
            validSizes{numSizes}=ones(h,w)/(h*w); % normalized averaging function
        end
    end
end


% Continue to add parts till model.numparts parts have been added

numAdded=0;
mid=rootSize(2)/2;
while numAdded<model.numparts

    for i=1:numSizes
        % Compute the normalized energy score of each valid 
        % part filter size at every location of the root
        convScore=conv2(EnergyMat,validSizes{i},'valid');
        copyScore=convScore;
        
        % Zero out scores for placement of parts that would lead to
        % overlapping symmetric parts
        Index=max(mid-rootSize(2)+2,1);
        convScore(:,Index:end)=-realmax;
        
        % Set all positions as temporarily invalid, if only one filter
        % remains to be placed. Later we will allow for centered placements
        % in addition to what is allowed at the end of this step. Therefore
        % at most one centered part can be placed.
        if (numAdded-model.numparts==1)
            convScore(:,:)=-Inf;
        end
        
        % Set scores for centered placement of parts to be the original
        % scores. Parts can only be centered if they have even width.
        
        if (mod(size(validSizes{i},2)/2,2)==0)
            convScore(:,mid-(size(validSizes{i},2)/2)+1)=copyScore(:,mid-(size(validSizes{i},2)/2)+1);
        end
        
        % Now find the maximum score for the shape
        
        [dummy, y_inds]=max(convScore);
        [maxSc, x]=max(dummy);
        y=y_inds(x);
        
        % Check if the placement x is centered
        sym_x=rootSize(2)+2-size(validSizes{i},2); % x for the symmetrically placed part
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
        model = updateModel(model, numAdded, rootWeights, x, y, validSizes, i, width, component);
        EnergyMat(y:y+size(validSizes{i},1)-1,x:x+size(validSizes{i},2)-1,:)=0;
        
        if ~centered
            numAdded=numAdded+1;
            model = updateModel(model, numAdded, rootWeights, sym_x, y, validSizes, i, width, component);
            EnergyMat(y:y+size(validSizes{i},1)-1,sym_x:sym_x+size(validSizes{i},2)-1,:)=0;
        end
    end
end
end
        

function model = updateModel(model, partIdx, rootWeights, x, y, validSizes, i, width, componentIdx)
cumIdx = length(model.partfilters) + 1;
model.partfilters{cumIdx}.w=rootWeights(y:y+size(validSizes{i},1)-1,x:x+size(validSizes{i},2)-1,:);
model.partfilters{cumIdx}.blocklabel = length(model.blocksizes)+1;
model.blocksizes(model.partfilters{cumIdx}.blocklabel) = width*size(validSizes{i},1)*31;
model.regmult(model.partfilters{cumIdx}.blocklabel) = 1;
model.learnmult(model.partfilters{cumIdx}.blocklabel) = 1;
model.lowerbounds{model.partfilters{cumIdx}.blocklabel} = -100*ones(model.blocksizes(model.partfilters{cumIdx}.blocklabel),1);
model.numblocks = model.numblocks+1;

model.defs{cumIdx}.anchor=[y x];
model.defs{cumIdx}.w=[0 0 1 1];                   % initially only allow quadratic terms
model.defs{cumIdx}.blocklabel = length(model.blocksizes)+1;
model.blocksizes(model.defs{cumIdx}.blocklabel) = numel(model.defs{cumIdx}.w);
model.regmult(model.defs{cumIdx}.blocklabel) = 1;
model.learnmult(model.defs{cumIdx}.blocklabel) = 1;
model.lowerbounds{model.defs{cumIdx}.blocklabel} = [-100, -100, 0, 0];
model.numblocks = model.numblocks+1;

model.components{componentIdx}.parts{partIdx}.partindex = cumIdx;
model.components{componentIdx}.parts{partIdx}.partidx = cumIdx;
model.components{componentIdx}.parts{partIdx}.defindex = cumIdx;
model.components{componentIdx}.parts{partIdx}.defidx = cumIdx;
end
        
        
        
        
        
    
   
