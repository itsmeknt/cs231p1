function overlap = computeOverlap(box1, box2)

intersection = zeros(1, 4);
intersection(1) = max(box1(1), box2(1));
intersection(2) = max(box1(2), box2(2));
intersection(3) = min(box1(3), box2(3));
intersection(4) = min(box1(4), box2(4));

intersectionArea = computeArea(intersection);
denominator = computeArea(box1)+computeArea(box2)-intersectionArea;
if denominator == 0
    overlap = 0;
else
    overlap = intersectionArea/denominator;
end

function area = computeArea(box)
width = box(3) - box(1);
height = box(4) - box(2);
area = max(0,width*height);