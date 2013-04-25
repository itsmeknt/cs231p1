% Compute the latent variables (root filter position,part positions and 
% model component to be used) and the overall score for every possible 
% position of the root filter in the feature pyramid
% 
% Inputs are:
% The input image features.
% The model (with root and part filters initialized)

function [component,rootLoc,partLoc,level,maxScore,Array]=latent(model,features,scales)

% The function returns the following:
% component: The model component (this is a latent variable)
% level: The pyramid level at which the best scoring root location is
% rootLoc: A 1x2 vector that gives the (x,y) location of the root at level
% partLoc: A 6x1 cell of 2x1 vectors giving the deformation of the parts:
%          partLoc{j} gives the distance of part j from its canonical
%          position at the scale (level-model.interval)
% maxScore: The overall score of the best latent position
% Array: The score array for every possible position of the root filter:
%        Array{1,k}[x,y] gives the score of placing the root filter at level
%        k and position (x,y) in the feature pyramid for the component i.

ScoreArray=cell(model.numcomponents,length(features));

% Find the original scale and last scale indices
orig_scale=find(scales==1);
last_scale=size(scales,1);

% we pad the feature maps to detect partially visible objects

for i=1:model.numcomponents
    
    firstTic = tic;
    % Get Model Rootfilter
    rootIdx=model.components{i}.rootindex;
    rootFilter=model.rootfilters{rootIdx}.w;
    rootSize=model.rootfilters{rootIdx}.size;
    
    rootfilters = cell(1,1);
    rootfilters{1} = rootFilter;
    conv_root=cell(last_scale-orig_scale+1,1);
    % Initialize the convolution of the root with each level
    for k=orig_scale:last_scale
        rFeatr = padarray(features{k}, [model.pady model.padx 0], 0);
        rResult=fconv(rFeatr,rootfilters,1,length(rootfilters)); % Compute the convolution
        conv_root{k}=rResult{1};
    end
    conv_part=cell(last_scale-orig_scale+1,model.numparts);
    deform=cell(model.numparts,1);
    for j=1:model.numparts
        % Get the part filters, part deformations and part anchors
        partIdx=model.components{i}.parts{j}.partindex;
        defIdx=model.components{i}.parts{j}.defindex;
        partFilter=model.partfilters{partIdx};
        partDef=model.defs{defIdx};
        
        
        % Compute the deformation cost matrix
        deform{j}=computeDefMatrix(rootSize,partDef);
        
        % Initialize the part convolution
        
        partfilters = cell(1,1);
        partfilters{1} = partFilter.w;
        % Convolve the part filters with the levels of the feature 
        % pyramid that are one octave below the root
        
        for k=orig_scale:last_scale
            pFeatr = padarray(features{k-model.interval}, [model.pady model.padx 0], 0);
            pResult=fconv(pFeatr,partfilters,1,length(partfilters)); % Compute the convolution  
            conv_part{k-model.interval,j}=pResult{1};
        end
    end
    
        firstTime = toc(firstTic)
    % Compute the score for each location of the root
    
    partTic = tic;
    for k=orig_scale:last_scale
        maxScore=-Inf;
        for y=1:size(features{k},1)-rootSize(1)+1
            for x=1:size(features{k},2)-rootSize(2)+1
                parts=cell(6,1);
                ScoreArray{i,k}(y,x)=conv_root{k}(y,x)+model.offsets{rootIdx}.w;
                partScore=cell(model.numparts,1);
                for j=1:model.numparts
                    if (x==57)
                        size(conv_part{k-model.interval,j});
                    end
                    [partScore{j},parts]=getScoreAndLoc(conv_part{k-model.interval,j},x,y,deform{j},rootSize);
                    ScoreArray{i,k}(y,x)=ScoreArray{i,k}(y,x)+partScore{j};
                   
                end
                if (ScoreArray{i,k}(y,x)>maxScore)
                    maxScore=ScoreArray{i,k}(y,x);
                    component=i;
                    rootLoc=[y x];
                    partLoc=parts;
                    level=k;
                end
            end
        end
    end
    partTocTime = toc(partTic)
end
Array=ScoreArray;

