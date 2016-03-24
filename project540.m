clear
load train.mat
load test.mat
X=train(:,1:end-1);
Y=train(:,end);
Xtest=test;
[x,y]=find(X>1000000000);
X(x,y)=0;
max(max(X))
