import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# from pandas.tools.plotting import scatter_matrix

from sklearn import tree
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge,Lasso,BayesianRidge,SGDRegressor,ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.feature_selection import RFECV
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.stats import randint

import seaborn as sns

root_dir="./"
N = 20000

train_data = pd.read_csv(root_dir+"train.csv")
test_data = pd.read_csv(root_dir+"test.csv")

train_data = train_data.drop(['url'], axis=1) #remove 'url' information.
train_data = train_data.drop(['timedelta'], axis=1) #remove 'url' information.

u = np.median(train_data['shares'])
s = np.std(train_data['shares'])
line=u+2*s
new_train_data=train_data[train_data['shares']<line]

X_train, X_val= train_test_split(new_train_data, test_size=0.2, random_state=42)

y_train = np.matrix(X_train['shares']).T
X_train = np.matrix(X_train.drop(['shares'], axis=1)) #Dropping both 'shares', the predicted variable and 'url', a text variable

y_val = np.matrix(X_val['shares']).T
X_val = np.matrix(X_val.drop(['shares'], axis=1))

#hyperparameter tuning
svrgs_parameters = {
    'kernel': ['rbf', 'poly'],
    'C':      [1.0, 2.0, 3.0, 6.0, 10],
    'gamma':  ['scale', 'auto'],
}

svr_cv = GridSearchCV(svm.SVR(), svrgs_parameters, cv=8, scoring= 'neg_mean_squared_error')
svr_cv.fit(X_train, y_train)
print("SVR GridSearch score: "+ str(svr_cv.best_score_))
print("SVR GridSearch params: " + str(svr_cv.best_params_))

model_svr_best = SVR(kernel='rbf', C=6, gamma='auto')
model_svr_best.fit(X_train, y_train)

final_prediction = model_svr_best.predict(X_val)
def MABS(vec1, vec2):
    return np.mean(np.abs(vec1 - vec2))

# reg = Lasso(alpha=10.0).fit(XTrain, yTrain)   #alpha: regularization strength
yHatTrain = model_svr_best.predict(X_train)
yHatVal = model_svr_best.predict(X_val)
print("Training error ", MABS(y_train, yHatTrain.T))
print("Validation error ", MABS(y_val, yHatVal.T))