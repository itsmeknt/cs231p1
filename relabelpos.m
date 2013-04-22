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
    detections = detect(feat, scale, model, -realmax);
    maxDetection = [];
    maxScore = -realmax;
    for j=1:length(detections)
       detection = detections(j); 
       if detection.score > maxScore
           maxDetection = [detection];
           maxScore = detection.score;
       elseif detection.score == maxScore
           maxDetection = [maxDetection; detection];
       end
    end
    
    % if theres a tie of multiple max score detection, comput best overlap
    % also remove detections with too little (<70%) overlap
    bestDetection = [];
    bestOverlap = -realmax;
    for j=1:length(maxDetection)
        detection = maxDetection(j);
        overlap = computeOverlap(detection.rootBbox, trueBbox);
        if overlap > .7 & overlap > bestOverlap
            bestOverlap = overLap;
            bestDetection = detection;
        end
    end
    
    newPos(1, i).x1 = bestDetection.rootBbox(1);
    newPos(1, i).y1 = bestDetection.rootBbox(2);
    newPos(1, i).x2 = bestDetection.rootBbox(3);
    newPos(1, i).y2 = bestDetection.rootBbox(4);
end
end