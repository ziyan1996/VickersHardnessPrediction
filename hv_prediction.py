"""Predict hardness values using XGBoost."""
from os.path import join
import xgboost as xgb
import numpy as np
import pandas as pd
from shaphypetune import BoostBoruta, BoostRFE

import matplotlib.pyplot as plt
from plotting import parity_with_err

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import uncertainty_toolbox as uct

from vickers_hardness.utils.uncertainty import log_cosh_quantile

recalibrate = True

# %% load dataset
X = pd.read_csv("hv_des.csv")
prediction = pd.read_csv("hv_comp_load.csv")
Y = prediction["hardness"]


# %% Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.9, test_size=0.1, random_state=100, shuffle=True
)

y_train = y_train.to_frame()
y_test = y_test.to_frame()

# %% scale the data
composition_train = X_train["composition"].to_frame()
composition_test = X_test["composition"].to_frame()
X_train.drop(columns=["composition"], inplace=True)
X_test.drop(columns=["composition"], inplace=True)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scl = scaler.transform(X_train)
X_test_scl = scaler.transform(X_test)


# %% XGB model construction
parameters = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=1000,
    verbosity=1,
    booster="gbtree",
    tree_method="auto",
    n_jobs=1,
    gamma=0.0001,
    min_child_weight=8,
    max_delta_step=0,
    subsample=0.6,
    colsample_bytree=0.7,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=4,
    scale_pos_weight=1,
    base_score=0.6,
    num_parallel_tree=1,
    importance_type="gain",
    eval_metric="rmse",
    nthread=4,
)

xgb_mdl = xgb.XGBRegressor(objective="reg:squarederror", **parameters)
xgb_hyp = BoostBoruta(xgb_mdl, importance_type="shap_importances")
xgb_hyp.fit(X_train_scl, y_train)

# %% uncertainty quantification
# https://towardsdatascience.com/confidence-intervals-for-xgboost-cac2955a8fde
alpha = 0.95
xgb_upper = xgb.XGBRegressor(objective=log_cosh_quantile(alpha), **parameters)
xgb_upp_hyp = BoostBoruta(xgb_upper, importance_type="shap_importances")
xgb_upp_hyp.fit(X_train_scl, y_train)
xgb_lower = xgb.XGBRegressor(objective=log_cosh_quantile(1 - alpha), **parameters)
xgb_low_hyp = BoostBoruta(xgb_lower, importance_type="shap_importances")
xgb_low_hyp.fit(X_train_scl, y_train)


# %% Prediction
y_pred = xgb_hyp.predict(X_test_scl)
y_upper = xgb_upp_hyp.predict(X_test_scl)
y_lower = xgb_low_hyp.predict(X_test_scl)

y_test_vals = y_test.values.ravel()

# https://handbook-5-1.cochrane.org/chapter_7/7_7_3_2_obtaining_standard_deviations_from_standard_errors_and.htm
y_std = (y_upper - y_lower) / 3.92  # hard-coded for 95% CI, sample_size is 1 (?)
if recalibrate:
    std_recalibrator = uct.recalibration.get_std_recalibrator(
        y_pred, y_std, y_test_vals, criterion="ma_cal"
    )
    y_std_calib = std_recalibrator(y_std)
else:
    y_std_calib = y_std

result_df = pd.DataFrame(
    {
        "actual_hardness": y_test_vals,
        "predicted_hardness": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "y_std": y_std,
        "y_std_calib": y_std_calib,
        "load": X_test["load"],
    }
)
result_df = composition_test.join(result_df)
result_df.to_csv("predicted_hv.csv", index=False)

parity_with_err(result_df, error_y="y_upper", error_y_minus="y_lower")
parity_with_err(result_df, error_y="y_std", fname="parity_err")
parity_with_err(result_df)

_, ax = plt.subplots(1, 1, figsize=(5, 5))
uct.viz.plot_calibration(y_pred, y_std, y_test_vals, ax=ax)
ax.set_title("")
plt.show()
plt.savefig(join("figures", "pre-cal.png"))

_, ax2 = plt.subplots(1, 1, figsize=(5, 5))
uct.viz.plot_calibration(y_pred, y_std_calib, y_test_vals, ax=ax2)
ax2.set_title("")
plt.show()
plt.savefig(join("figures", "post-cal.png"))

print("MAE: ", mean_absolute_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))
print("A file named predicted_hv.csv has been generated.\nPlease check your folder.")

# %% Code Graveyard
