function [ fid ] = cacheFeaturePyramids(posOrNegs, sbin, interval)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dirPath = 'cache/';
if ~exist(dirPath, 'file')
    mkdir(dirPath);
end

for i=1:length(posOrNegs)
    id = posOrNegs(i).id;
    if iscell(id)
        id = id{1};
    end
    if exist([dirPath id '-feat-r.mat'], 'file')
        continue;
    end
    
    im = color(imread(posOrNegs(i).im));
    [feat scale] = featpyramid(im(:,end:-1:1,:), sbin, interval);
    save([dirPath id '-feat-r'], 'feat');
    save([dirPath id '-scale'], 'scale');
end


end

