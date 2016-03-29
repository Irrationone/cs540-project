project540
%Train model
model = logRegL1(X,y,0.1);

%Sparsity of w
numberOfNonZero = nnz(model.w)

%Training error
yhat = model.predict(model,X);
trainingError = sum(yhat ~= y)/n
