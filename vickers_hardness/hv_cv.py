"""Predict hardness values using XGBoost."""
from os.path import join
import numpy as np
import pandas as pd

from vickers_hardness.utils.plotting import parity_with_err

from sklearn.model_selection import (
    KFold,
    GroupKFold,
    cross_validate,
)

from sklearn.metrics import mean_absolute_error, mean_squared_error

from vickers_hardness.vickers_hardness_ import VickersHardness

recalibrate = True
split_by_groups = True

# %% load dataset
X = pd.read_csv(join("vickers_hardness", "data", "hv_des.csv"))
prediction = pd.read_csv(join("vickers_hardness", "data", "hv_comp_load.csv"))
y = prediction["hardness"]

# %% K-fold cross-validation
if split_by_groups:
    cv = GroupKFold()
else:
    cv = KFold(shuffle=True, random_state=100)  # ignores groups

results = cross_validate(
    VickersHardness(hyperopt=True),
    X,
    y,
    groups=X["composition"],
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_estimator=True,
)

estimators = results["estimator"]
result_dfs = [estimator.result_df for estimator in estimators]
merge_df = pd.concat(result_dfs)
merge_df["actual_hardness"] = y

parity_with_err(
    merge_df, error_y="y_upper", error_y_minus="y_lower", fname="parity_ci_cv"
)
parity_with_err(merge_df, error_y="y_std", fname="parity_stderr_cv")
parity_with_err(merge_df, fname="parity_stderr_calib_cv")

y_true, y_pred = [merge_df["actual_hardness"], merge_df["predicted_hardness"]]
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
# CV-MAE: 2.2957275 (HV)
# CV-RMSE: 3.4197476 (HV)

merge_df.sort_index().to_csv(join("results", "cv-results.csv"))
1 + 1


# %% Code Graveyard
