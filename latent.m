% Compute the latent variables (root filter position,part positions and 
% model component to be used) and the overall score for every possible 
% position of the root filter in the feature pyramid
% 
% Inputs are:
% The input image features.
% The model (with root and part filters initialized)

function [component,rootLoc,partLoc,level,maxScore,ScoreMatrix]=latent(model,features,scales)

% The function returns the following:
% component: The model component (this is a latent variable)
% level: The pyramid level at which the best scoring root location is
% rootLoc: A 1x2 vector that gives the (x,y) location of the root at level
% partLoc: A 6x2 matrix giving the deformation of the parts:
%          partLoc(j,:) gives the distance of part j from its canonical
%          position at the scale (level-model.interval)
% maxScore: The overall score of the best latent position
% Array: The score array for every possible position of the root filter:
%        Array{1,k}[x,y] gives the score of placing the root filter at level
%        k and position (x,y) in the feature pyramid for the component i.

ScoreMatrix=cell(model.numcomponents,length(features));

% Find the original scale and last scale indices
orig_scale=find(scales==1);
last_scale=size(scales,1);
numScales = last_scale-orig_scale+1;

% initialize fconv variables
rootfilters = cell(1, length(model.rootfilters));
for i=1:length(model.rootfilters)
    rootfilters{i} = model.rootfilters{i}.w;
end
partfilters = cell(1, length(model.partfilters));
for i=1:length(model.partfilters)
    partfilters{i} = model.partfilters{i}.w;
end

maxScore = -realmax;

% fconv after padding
conv_roots = cell(length(scales), length(model.rootfilters));
conv_parts = cell(length(scales), length(model.partfilters));
for k=1:length(scales)
    featuresPadded = padarray(features{k}, [model.pady model.padx 0], 0);
    
    if k > model.interval
        conv_roots(k, :) = fconv(featuresPadded, rootfilters, 1, length(rootfilters));    
    end
    if k <= length(scales) - model.interval
        conv_parts(k, :) = fconv(featuresPadded, partfilters, 1, length(partfilters));
    end
end

for i=1:model.numcomponents
    rootindex = model.components{i}.rootindex;
    rootsize = model.rootfilters{rootindex}.size;
    
    % initialize deformation matrix
    deform=zeros(2*rootsize(1),2*rootsize(2),model.numparts);       % tensor vector
    for j=1:model.numparts
        % Compute the deformation cost matrix
        defIdx=model.components{i}.parts{j}.defindex;
        partDef=model.defs{defIdx};
        deform(:,:,j)=computeDefMatrix(rootsize,partDef);
    end 
    
    % Compute the score for each location of the root
    % parallelize
    %matlabpool open feature('numcores');
    for k=orig_scale:last_scale
        rootScoreMatrix = conv_roots{k, model.components{i}.rootindex};
        if size(rootScoreMatrix,1)-2*model.pady <= 0 || size(rootScoreMatrix,2)-2*model.padx <= 0
            ScoreMatrix{i, k} = zeros(0,0);
            continue;
        end
        
        ScoreMatrix{i, k} = rootScoreMatrix(1+model.pady:end-model.pady,1+model.padx:end-model.padx) + model.offsets{rootindex}.w;
        
        convPartSize = size(conv_parts{k-model.interval, 1});
        partConvTensors=zeros(convPartSize(1), convPartSize(2), model.numparts);
        for j=1:model.numparts
            partindex = model.components{i}.parts{j}.partindex;
            partConvTensors(:,:,j) = conv_parts{k-model.interval, partindex};
        end
        
        for x=1:size(features{k},2)-rootsize(2)+1
            for y=1:size(features{k},1)-rootsize(1)+1
                partScores = partConvTensors((2*y):(2*y+2*rootsize(1)-1),(2*x):(2*x+2*rootsize(2)-1),:) + deform;
                
                [max_xs,ind_xs]=max(partScores, [], 2);
                [bestPartScores,partLocsY]=max(max_xs, [], 1);
                
                partLocs = zeros(model.numparts, 2);
                for p=1:model.numparts
                    partLocs(p,1) = partLocsY(1,1,p);
                    partLocs(p,2) = ind_xs(partLocsY(1,1,p), 1, p);
                end
                ScoreMatrix{i,k}(y,x)=ScoreMatrix{i,k}(y,x)+sum(bestPartScores);
                
                if (ScoreMatrix{i,k}(y,x)>maxScore)
                    maxScore=ScoreMatrix{i,k}(y,x);
                    component=i;
                    rootLoc=[y x];
                    partLoc=partLocs;
                    level=k;
                end
            end
        end
    end
end


