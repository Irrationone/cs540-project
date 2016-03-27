% clear
% load train.mat
% load test.mat
[n,d]=size(train(:,1:end-1));
[nt,dt]=size(test);

%%Combining for standardization
Xc=[train(:,1:end-1);test];

%%Standardizing and removing duplicates
Xc=unique(Xc','rows','stable')';
[Xc,mu,sigma] = standardizeCols(Xc);

%%Seperating again
X=Xc(1:n,:);
Xtest=Xc(n+1:end,:);


y=train(:,end);
%Train model
model = logRegL0(X,y,.5);

%Sparsity of w
numberOfNonZero = nnz(model.w)

%Training error
yhat = model.predict(model,X);
trainingError = sum(yhat ~= y)/n



%Predictions
yhat = model.predict(model,Xtest);
