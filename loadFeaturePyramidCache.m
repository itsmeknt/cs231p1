function [ feat scale ] = loadFeaturePyramidCache( id )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

dirPath = 'cache/';

feat = [];
scales = [];

if exist(dirPath, 'file')
    if iscell(id)
        id = id{1};
    end
    if iscell(id)
        id = id{1};
    end
    loadedFeat = load([dirPath id '-feat']);
    feat = loadedFeat.feat;
    
    loadedScale = load([dirPath id '-scale']);
    scale = loadedScale.scale;
end

end

