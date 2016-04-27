import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing

train = pd.DataFrame.from_csv('../../train.csv')
test = pd.DataFrame.from_csv('../../test.csv')

## Dimensions of train set
ntrain,dtrain = train.shape

## Dimensions of test set
ntest, dtest = test.shape

X = train.drop(['TARGET'], axis=1)
Xtest = test

## Standardize sets together
Xtotal = X.append(Xtest)
Xtotal_scaled = preprocessing.scale(Xtotal)
X_scaled,Xtest_scaled = np.split(Xtotal_scaled, [ntrain])

targets = np.array(train.TARGET)

from sklearn import cross_validation

sss = cross_validation.StratifiedShuffleSplit(targets, 1, test_size=0.2, random_state=0)
(train_index,valid_index) = list(sss)[0]

X_train = X_scaled[train_index,]
X_valid = X_scaled[valid_index,]
y_train = targets[train_index]
y_valid = targets[valid_index]

print(X_train.shape)
print(X_valid.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

forest = RandomForestClassifier(n_estimators = 300, class_weight="balanced", bootstrap=False)
forest = forest.fit(X_train, y_train)

y_hat = forest.predict(X_valid)
y_hat2 = forest.predict(X_train)
print("Training accuracy:",forest.score(X_train, y_train))
print("Validation accuracy:",forest.score(X_valid, y_valid))
print("Training AUC:",roc_auc_score(y_train, y_hat2))
print("Validation AUC:",roc_auc_score(y_valid, y_hat))
