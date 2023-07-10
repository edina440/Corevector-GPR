# Corevector-GPR
GPR combined with corevector regresssion
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:47:42 2023

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import os
os.chdir("C://Users//HP//Desktop//paper 1//all analysis docs//sun interactions")


unv_hour = pd.read_csv("lasso_interaction_sun py.csv")
unv_hour_df = unv_hour[['GHI','AirTC_Avg','WS_ms_S_WVT','WindDir_D1_WVT','WindDir_SD1_WVT','BP_mB_Avg','lag1','lag2','AirTC_Avg.1','RH.1','WS_ms_S_WVT.1','WindDir_SD1_WVT.1','BP_mB_Avg.1','lag1.1','lag2.1','WS_ms_S_WVTAirTC_Avg','WindDir_D1_WVTAirTC_Avg','BP_mB_AvgAirTC_Avg','noltrendAirTC_Avg','lag1AirTC_Avg','lag2AirTC_Avg','WS_ms_S_WVTRH','WindDir_D1_WVTRH','WindDir_SD1_WVTRH','noltrendRH','WindDir_D1_WVTWS_ms_S_WVT','WindDir_SD1_WVTWS_ms_S_WVT','BP_mB_AvgWS_ms_S_WVT','noltrendWS_ms_S_WVT','lag1WS_ms_S_WVT','lag2WS_ms_S_WVT','WindDir_SD1_WVTWindDir_D1_WVT','BP_mB_AvWindDir_D1_WVT','noltrendWindDir_D1_WVT','lag1WindDir_D1_WVT','lag2WindDir_D1_WVT','BP_mB_Avg.2','lag1.2','lag2.2','noltrendBP_mB_Avg','lag1BP_mB_Avg','lag2BP_mB_Avg','noltrendBP_mB_Avg.1','lag1BP_mB_Avg.1','lag2BP_mB_Avg.1','lag1noltrend','lag2noltrend','lag2lag1']]
unv_hour_df_copy = unv_hour_df.copy()
test_set = unv_hour_df_copy.sample(frac=0.80, random_state=0)
train_set = unv_hour_df_copy.drop(test_set.index)

train_set = np.array(train_set)
y_train = train_set[:,0]
X_train = train_set[:,1:48]

test_set = np.array(test_set)
y_test = test_set[:,0]
X_test = test_set[:,1:48]
#############################  define the kernels  #########################################################

Matern = 10.0 * Matern(length_scale=0.5)
RBF    = 10.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 11.0))
RationalQuadratic = 10.0 * RationalQuadratic(length_scale=1.9, alpha=0.1)

DotProduct = 1.0*DotProduct(sigma_0=10)
###########################  fit the gaussian model   ###################


# Specify Gaussian Process for matern kernel
gprM = GaussianProcessRegressor(kernel=Matern).fit(X_train, y_train)
y_predM,y_stdM=gprM.predict(X_test,return_std=True)

# Specify Gaussian Process for RBF kernel
gprR = GaussianProcessRegressor(kernel=RBF).fit(X_train, y_train)
y_predR,y_stdR=gprR.predict(X_test,return_std=True)

# Specify Gaussian Process for RationalQuadratic kernel
gprRQ = GaussianProcessRegressor(kernel=RationalQuadratic).fit(X_train, y_train)
y_predRQ,y_stdRQ=gprRQ.predict(X_test,return_std=True)



gprDot = GaussianProcessRegressor(kernel=DotProduct, alpha=1e-1).fit(X_train, y_train)
y_predDot,y_stdDot=gprDot.predict(X_test,return_std=True)
 

##########################   Minimum enclosed ball for GP   #################################################

import math
import random
import miniball

# Specify the Minimum enclosed ball for GP(matern)
arr_M = np.vstack((y_predM, X_test[:,0]))
xy_M = np.transpose(arr_M)
C_M, R2_M = miniball.get_bounding_ball(xy_M)
print('Center', C_M)
print('Radius', math.sqrt(R2_M))


# Specify the Minimum enclosed ball for GP(RBF)
arr_R = np.vstack((y_predR, X_test[:,0]))
xy_R = np.transpose(arr_R)
C_R, R2_R = miniball.get_bounding_ball(xy_R)
print('Center', C_R)
print('Radius', math.sqrt(R2_R))


# Specify the Minimum enclosed ball for GP(RationalQuadratic)
arr_RQ = np.vstack((y_predRQ, X_test[:,0]))
xy_RQ = np.transpose(arr_RQ)
C_RQ, R2_RQ = miniball.get_bounding_ball(xy_RQ)
print('Center', C_RQ)
print('Radius', math.sqrt(R2_RQ))




# Specify the Minimum enclosed ball for GP(DotProduct)
arr_Dot = np.vstack((y_predDot, X_test[:,0]))
xy_Dot = np.transpose(arr_Dot)
C_Dot, R2_Dot = miniball.get_bounding_ball(xy_Dot)
print('Center', C_Dot)
print('Radius', math.sqrt(R2_Dot))
###########################   extracting forecasts    #########################

prediction = pd.DataFrame(y_predRQ, columns=['predictions']).to_csv('prediction.csv')

###########################   model accuracy     ##############################################################

# using the chosen model with the min radius, we now check its accuracy and plot a 
# real time plot of the pred vs actual values

###RSME
from math import sqrt
from sklearn.metrics import mean_squared_error

mseM = mean_squared_error(y_test, y_predM)
rmseM = sqrt(mseM)
print('Matern RMSE: %f' % rmseM)

mseR = mean_squared_error(y_test, y_predR)
rmseR = sqrt(mseR)
print('Matern RMSE: %f' % rmseR)

mseRQ = mean_squared_error(y_test, y_predRQ)
rmseRQ = sqrt(mseRQ)
print('Matern RMSE: %f' % rmseRQ)

mseDot = mean_squared_error(y_test, y_predDot)
rmseDot = sqrt(mseDot)
print('Matern RMSE: %f' % rmseDot)

##MAE

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_predM)
mean_absolute_error(y_test, y_predR)
mean_absolute_error(y_test, y_predRQ)
mean_absolute_error(y_test, y_predDot)

#MAPE
import numpy as np
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

mape(y_test, y_predDot)
mape(y_test, y_predM)
mape(y_test, y_predR)
mape(y_test, y_predRQ)



###########################  Feature Importance                               #################################

from xgboost import plot_importance

# feature importance
print(predRQ.feature_importances_)
# plot
pyplot.bar(range(len(gprM.feature_importances_)),gprM.feature_importances_)
pyplot.show()

plot_importance(gprM)
pyplot.show()





###########################  static plot of the observed vs predicted values  ##################################

#use the minimum radius GPR to plot the predicted vs the observed values
#predicted(y_pred) vs observed values(y_test)
y_pred_RQ, = plt.plot(y_predRQ, label='y_pred',linewidth=0.5)
"linewidth=0.10"
y_test_RQ, = plt.plot(y_test, label='y_test',linewidth=0.5)
plt.legend(handles=[y_pred_RQ, y_test_RQ])
plt.title("GPR with interaction for SUN")
plt.xlabel('Time(hours)')
plt.ylabel('GHI')
plt.show()
###########################  shorter time frame

#use the minimum radius GPR to plot the predicted vs the observed values
#predicted(y_pred) vs observed values(y_test)
y_pred_RQ, = plt.plot(y_predRQ, label='y_pred',linewidth=0.5)
y_test_RQ, = plt.plot(y_test, label='y_test',linewidth=0.5)
plt.legend(handles=[y_pred_RQ, y_test_RQ])
plt.title("GPR without interactions for SUN")
plt.xlabel('Time(hours)')
plt.ylabel('GHI')
plt.xlim([0, 150])
plt.show()
