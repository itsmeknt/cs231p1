function [scores,positions]=TwoDDistTransform(conv,params)


% Two dimensional distance transform function. signal is a two dimensional
% sampled signal. params are the quadratic deformation features. scores
% gives the distance transform at each location of the signal. position
% gives the corresponding placement of the part.

[m,n]=size(conv);   % size of convolution matrix
a2=params(1);       % quadratic coefficient for squared distance in y
b2=params(2);       % quadratic coefficient for linear distance in y
a1=params(3);       % quadratic coefficient for squared distance in x
b1=params(4);       % quadratic coefficient for linear distance in x

x_positions=zeros(m,n);
dummyPos=zeros(n,m);
dummyScores=zeros(n,m);
scores=zeros(m,n);
for j=1:n
    [dummyScores(j,:) dummyPos(j,:)]=OneDDT(conv(:,j)',[a1 b1]);
end
positions=zeros(m,n,2);
dummyScores=dummyScores';
dummyPos=dummyPos';
for i=1:m
    [scores(i,:) x_positions(i,:)]=OneDDT(dummyScores(i,:),[a2 b2]);
    positions(i,:,1)=dummyPos(i,x_positions(i,:));
    positions(i,:,2)=x_positions(i,:);
end
