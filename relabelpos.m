function [newPos] = relabelpos(pos, model)

% Relabel positive examples given a trained root filter

filt_px_width=model.sbin*model.maxsize(2);
filt_px_height=model.sbin*model.maxsize(1);
newPos = pos;
for i=1:size(pos,2)
    % Existing bounding box and image size
    trueBbox = [pos(1,i).x1, pos(1,i).y1, pos(1,i).x2, pos(1,i).y2];
    bbox_width=trueBbox(3) - trueBbox(1);
    bbox_height=trueBbox(4) - trueBbox(2);
    image=imread(pos(1,i).im);
    image = color(image);
    image_size=size(image);
    
    % pad the existing bounding box
    padx=ceil(0.3*min(bbox_width,filt_px_width));
    pady=ceil(0.3*min(bbox_height,filt_px_height));
    
    % Compute range over which the root filter latent position is searched
    new_x1=max(1,pos(1,i).x1-padx);
    new_x2=min(image_size(2),pos(1,i).x2+padx);
    new_y1=max(1,pos(1,i).y1-pady);
    new_y2=min(image_size(1),pos(1,i).y2+pady);
    
    % Construct the padded image and clear the original image
    testim=image(new_y1:new_y2,new_x1:new_x2,:);
    clear image;
    
    % Compute Score
    [feat, pyramidScales] = featpyramid(input, model.sbin, model.interval);
    [detectionsAboveThreshold, dummy, dummy2] = detect(feat, scale, model);
    [scores,scales,padx_feature,pady_feature]=computeScores(testim,model,false);
    maxScore = -realmax;
    maxBbox = [-1, -1, -1, -1];
    for c=1:model.numcomponents
        scorePyramid = scores{c};
        rsize = model.rootfilters{model.components{c}.rootindex}.size;
        for level=1:length(scorePyramid)
            score = scorePyramid{level};
            scale = scales(level);
            
            x = -1;
            y = -1;
            while true
                [maxCols, ys] = max(score);
                [maxVal, xs] = max(maxCols);
                if (maxVal <= maxScore)
                    break;
                end
                
                x = xs;
                y = ys(x);
                
                predictedBbox = getBoundingBox(x, y, scale, padx_feature, pady_feature, rsize);
                overlap = computeOverlap(predictedBbox, trueBbox);
                if overlap < 0.7
                    score(y, x) = -realmax;
                    x = -1;
                    y = -1;
                else
                    maxScore = maxVal;
                    maxBbox = predictedBbox;
                    break;
                end
            end
        end
    end
    newPos(1, i).x1 = maxBbox(1);
    newPos(1, i).y1 = maxBbox(2);
    newPos(1, i).x2 = maxBbox(3);
    newPos(1, i).y2 = maxBbox(4);
end
end