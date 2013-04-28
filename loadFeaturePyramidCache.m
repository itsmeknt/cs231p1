function [ feat scale ] = loadFeaturePyramidCache( id, im, sbin, interval )
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
    
    try
        loadedFeat = load([dirPath id '-feat']);
        feat = loadedFeat.feat;
        
        loadedScale = load([dirPath id '-scale']);
        scale = loadedScale.scale;
    catch
        im = color(imread(im));
        [feat scale] = featpyramid(im, sbin, interval);
        dirPath = 'cache/';
        if ~exist(dirPath, 'file')
            mkdir(dirPath);
        end
        if ~exist([dirPath id '-feat.mat'], 'file')
            save([dirPath id '-feat'], 'feat');
            save([dirPath id '-scale'], 'scale');
        end
    end
end

end

