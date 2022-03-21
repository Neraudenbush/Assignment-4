# Assignment-4
```Python
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
import xgboost as xgb
```
```Python
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
  ```
  ```Python
  #Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```
```Python
data = pd.read_csv('concrete.csv')
X = data[['cement','slag','ash','water','superplastic','coarseagg','fineagg','age']].values
y = data['strength'].values
```
Now we must decide what regressors to test for boosting. I decided to test random forest, decision tree and linear model.
```Python
X = data[['cement','slag','ash','water','superplastic','coarseagg','fineagg','age']].values
y = data['strength'].values
model_boosting1 = RandomForestRegressor(n_estimators=100,max_depth=3)
from sklearn import linear_model as lm
model_boosting2 = lm.LinearRegression()
model_boosting3 = tree(max_depth = 5)
```
Now we test the boosters with a 5 fold cross validation
(I cut down on the validation because it was taking a very long time to run on my computer)
```Python
mse_rf = []
mse_lm = []
mse_tree = []

kf = KFold(n_splits=5,shuffle=True,random_state=410)
  # this is the Cross-Validation Loop
for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
    yhat_rf = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting1,2)
    mse_rf.append(mse(ytest,yhat_rf))
    yhat_lm = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting2,2)
    mse_lm.append(mse(ytest,yhat_lm))
    yhat_tree = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting3,2)
    mse_tree.append(mse(ytest,yhat_tree))
    print('checkpoint')
print('The Cross-validated Mean Squared Error for LWR Boosted with Random Forest is : '+str(np.mean(mse_rf)))
print('The Cross-validated Mean Squared Error for LWR Boosted with Linear Model is : '+str(np.mean(mse_lm)))
print('The Cross-validated Mean Squared Error for LWR Boosted with Decision Tree is : '+str(np.mean(mse_tree)))
```
Results
```Markdown

```

LightGBM is (Light Gradient Boosting Machine) is a method of gradient boosting that is focused on preformance and scalability. It is very similar to XGB except for the way that the decision tree is constructed. In XGB the tree's are constructed row by row but in LightGBM constructs the tree by chosing leaves that the program belive will minimize the amount of loss. LightGBM uses a histogram style decision tree to help lower memory consuption and improve efficiency. I assume this is why it is called "light".

Lets test it
```Python
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 1000
}

lgbm = lgb.LGBMRegressor(**hyper_params)
```
```Python
mse_lgbm = []

for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)

    lgbm.fit(xtrain, ytrain,
        eval_set=[(xtest, ytest)],
        eval_metric='l1',
        early_stopping_rounds=1000)
    y_pred = lgbm.predict(xtest, num_iteration=lgbm.best_iteration_)
    mse_lgbm.append(mse(ytest,y_pred))

    
print('The Cross-validated Mean Squared Error for LightGBM is : '+str(np.mean(mse_lgbm)))
```
Results
```Markdown
The Cross-validated Mean Squared Error for LightGBM is : 23.578002282665324
