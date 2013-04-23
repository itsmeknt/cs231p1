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

maxScore=-Inf;
ScoreArray=cell(model.numcomponents,size(features));
for i=1:model.numcomponents
    
    % Get Model Rootfilter
    rootIdx=model.components{i}.rootindex;
    rootFilter=model.rootfilters{rootIdx}.w;
    rootSize=model.rootfilters{rootIdx}.size;
    
    % Find the original scale and last scale indices
    orig_scale=find(scales==1);
    last_scale=size(scales,1);
    
    % Initialize the convolution of the root with each level
    conv_root=cell(last_scale-orig_scale+1);
    for k=orig_scale:scales(end)
            conv_root{k}=imfilter(features{k},rootFilter); % Compute the convolution
    end
    
    deform=cell(model.numparts);
    for j=1:model.numparts
        % Get the part filters, part deformations and part anchors
        partIdx=model.components{i}.parts{j}.partindex;
        defIdx=model.components{i}.parts{j}.defindex;
        partFilter=model.partfilters{partIdx};
        partDef=model.defs{defIdx};
        
        
        % Compute the deformation cost matrix
        deform{j}=computeDefMatrix(rootSize,partDef);
        
        
        % Initialize the part convolution
        conv_part=cell(last_scale-orig_scale+1,model.numparts);
        % Convolve the part filters with the levels of the feature 
        % pyramid that are one octave below the root

        for k=(orig_scale):(scales(end))
                conv_part{k,j}=imfilter(features{k-model.interval},partFilter); % Compute the convolution 
        end   
    end
    
    % Compute the score for each location of the root
    
    for k=orig_scale:scales(end)
        for y=1:size(features{k},1)
            for x=1:size(features{k},2)
                parts=cell(6);
                ScoreArray{i,k}(y,x)=conv_root{k}(y,x)+model.offsets{rootIdx}.w;
                partScore=cell(model.numparts);
                for j=1:model.numparts
                    [partScore{j},parts]=getScoreAndLoc(conv_part{k,j},x,y,deform{j},rootSize);
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
    
end
    
Array=ScoreArray{component,:};

