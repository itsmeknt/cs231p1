function [scores,positions]=distTransform(signal,params)

% One dimensional distance transform function. signal is a one dimensional
% sampled signal. params are the quadratic deformation features. scores
% gives the distance transform at each location of the signal. position
% gives the corresponding placement of the part.

n=length(signal);   % no. of points in the signal
a=params(1);        % quadratic coefficient for squared distance
b=params(2);        % quadratic coefficient for linear distance

v=zeros(n,1);
z=zeros(n+1,1);
v(1)=1;
k=1;
z(1)=-Inf;
i=2;

while (i<=n)
    % compute intersection of parabola from i with parabola from v(k)
    s=(-a*(i^2-v(k)^2)+signal(i)-signal(v(k))+b*(i-v(k)))/(-2*a*(i-v(k)));
    if (s>z(k)) % Intersection is to the right of the previous change pt.
        k=k+1;  % Add current parabola to envelope
        z(k)=s; % Update rightmost intersection point
        v(k)=i; % Update right most parabola index
        i=i+1;  % Increment parabola index
    else
        k=k-1;  % Decrement k to compare with previous parabola in envelope
    end
end
z(k+1)=Inf;     % Last parabola is minimal till infinity
v=v(1:k);       % Truncate number of parabolas
z=z(1:k+1);     % Truncate parabola ranges

scores=zeros(n,1);
positions=zeros(n,1);
for i=1:n
    q=find((z>=i),1,'first');  % compute the parabola index for given domain point
    % Get score and position
    scores(i,1)=-(a*(i-v(q-1))^2+b*(i-v(q-1)))+signal(v(q-1));
    positions(i,1)=v(q-1);
end




