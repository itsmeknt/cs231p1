function [scores,positions]=TwoDDistTransform(conv,params)


% Two dimensional distance transform function. signal is a two dimensional
% sampled signal. params are the quadratic deformation features. scores
% gives the distance transform at each location of the signal. position
% gives the corresponding placement of the part.

[m,n]=size(conv);   % size of convolution matrix
a2=params(1);       % quadratic coefficient for linear distance in y
b2=params(2);       % quadratic coefficient for linear distance in x
a1=params(3);       % quadratic coefficient for squared distance in y
b1=params(4);       % quadratic coefficient for linear distance in x

x_positions=zeros(n,m);
dummyPos=zeros(m,n);
dummyScores=zeros(m,n);
scores=zeros(n,m);
for j=1:n
    [dummyScores(:,j) dummyPos(:,j)]=distTransform(conv(:,j),[a2 b2]);
end
positions=zeros(m,n,2);
for i=1:m
    [scores(:,i) x_positions(:,i)]=distTransform(dummyScores(i,:),[a1 b1]);
    positions(i,:,2)=x_positions(:,i)';
    positions(i,:,1)=dummyPos(i,x_positions(:,i)');
end

scores=scores';