# -*- coding: utf-8 -*-
"""

@author: Ziyan Zhang, University of Houston
"""

# Call functions
import xgboost as xgb
import numpy as np
import pandas as pd
from pandas import read_excel
from sklearn.utils import resample
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# load dataset
DE = read_excel('hv_des.xlsx')
array = DE.values
X = array[:,1:142]
Y = array[:,0]


# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1,random_state=0, shuffle=True)

#scale the data

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# XGB model construction
xgb_model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=300, verbosity=1, objective='reg:squarederror',
                             booster='gbtree', tree_method='auto', n_jobs=1, gamma=0.0001, min_child_weight=8,max_delta_step=0,
                             subsample=0.6, colsample_bytree=0.7, colsample_bynode=1, reg_alpha=0,
                             reg_lambda=4, scale_pos_weight=1, base_score=0.6, missing=None,
                             num_parallel_tree=1, importance_type='gain', eval_metric='rmse',nthread=4).fit(X_train,y_train)

# Prediction
prediction = pd.read_excel('pred_hv_descriptors.xlsx')
a = prediction.values
b = a[:,1:142]
result=xgb_model.predict(b)
composition=pd.read_excel('pred_hv_descriptors.xlsx',sheet_name='Sheet1', usecols="A")
composition=pd.DataFrame(composition)
result=pd.DataFrame(result)
predicted=np.column_stack((composition,result))
predicted=pd.DataFrame(predicted)
predicted.to_excel('predicted_hv.xlsx', index=False, header=("Composition","Predicted Vickers hardness"))
print("A file named predicted_hv.xlsx has been generated.\nPlease check your folder.")
