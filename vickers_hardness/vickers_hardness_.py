from os.path import join
from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd
from shaphypetune import BoostBoruta, BoostRFE
from sklearn.base import BaseEstimator

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import preprocessing

import uncertainty_toolbox as uct

from vickers_hardness.utils.uncertainty import log_cosh_quantile


class VickersHardness(BaseEstimator):
    def __init__(self, recalibrate=True, hyperopt=True, xgb_parameters=None):
        self.recalibrate = recalibrate
        self.hyperopt = hyperopt
        if xgb_parameters is None:
            self.xgb_parameters = dict(
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
        else:
            self.xgb_parameters = xgb_parameters

    def get_params(self, deep=True):
        if not deep:
            return {"recalibrate": self.recalibrate, "hyperopt": self.hyperopt}
        else:
            return {
                "recalibrate": self.recalibrate,
                "hyperopt": self.hyperopt,
                "xgb_parameters": self.xgb_parameters,
            }

    def fit(self, X_train, y_train, hyperopt=None):
        if hyperopt is None:
            hyperopt = self.hyperopt

        self.X_train = X_train
        self.y_train = y_train
        # %% scale the data
        self.composition_train = self.X_train["composition"].to_frame()
        self.X_train.drop(columns=["composition"], inplace=True)
        self.scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scl = self.scaler.transform(X_train)

        # %% XGB model construction
        self.xgb_mdl = xgb.XGBRegressor(
            objective="reg:squarederror", **self.xgb_parameters
        )
        if hyperopt:
            self.xgb_mdl = BoostBoruta(self.xgb_mdl, importance_type="shap_importances")
        self.xgb_mdl.fit(X_train_scl, y_train)

        # %% uncertainty quantification
        # https://towardsdatascience.com/confidence-intervals-for-xgboost-cac2955a8fde
        alpha = 0.95
        self.xgb_upper = xgb.XGBRegressor(
            objective=log_cosh_quantile(alpha), **self.xgb_parameters
        )
        self.xgb_lower = xgb.XGBRegressor(
            objective=log_cosh_quantile(1 - alpha), **self.xgb_parameters
        )

        if hyperopt:
            self.xgb_upper = BoostBoruta(
                self.xgb_upper, importance_type="shap_importances"
            )
            self.xgb_lower = BoostBoruta(
                self.xgb_lower, importance_type="shap_importances"
            )

        self.xgb_upper.fit(X_train_scl, y_train)
        self.xgb_lower.fit(X_train_scl, y_train)

    def predict(self, X_test, y_test=None, verbose=True):
        if y_test is None:
            skip_mae = True
            y_test = np.zeros(X_test.shape[0])
        else:
            skip_mae = False

        # %% Prediction
        self.X_test = X_test
        self.composition_test = self.X_test["composition"].to_frame()
        self.X_test.drop(columns=["composition"], inplace=True)
        X_test_scl = self.scaler.transform(X_test)

        self.y_pred = self.xgb_mdl.predict(X_test_scl)
        self.y_upper = self.xgb_upper.predict(X_test_scl)
        self.y_lower = self.xgb_lower.predict(X_test_scl)

        if isinstance(y_test, pd.Series):
            y_test_vals = y_test.values.ravel()
        elif isinstance(y_test, np.ndarray):
            y_test_vals = y_test.ravel()
        elif isinstance(y_test, list):
            y_test_vals = y_test
        else:
            raise NotImplementedError("y_test should be a Series, ndarray, or list")

        # https://handbook-5-1.cochrane.org/chapter_7/7_7_3_2_obtaining_standard_deviations_from_standard_errors_and.htm
        # hard-coded for 95% CI, sample_size is 1 (?)
        self.y_std = (self.y_upper - self.y_lower) / 3.92
        self.y_std[self.y_std <= 0] = 1e-4
        if self.recalibrate:
            self.std_recalibrator = uct.recalibration.get_std_recalibrator(
                self.y_pred, self.y_std, y_test_vals, criterion="ma_cal"
            )
            self.y_std_calib = self.std_recalibrator(self.y_std)
        else:
            self.y_std_calib = self.y_std

        self.test_load = X_test["load"]
        result_df = pd.DataFrame(
            {
                "actual_hardness": y_test_vals,
                "predicted_hardness": self.y_pred,
                "y_lower": self.y_lower,
                "y_upper": self.y_upper,
                "y_std": self.y_std,
                "y_std_calib": self.y_std_calib,
                "load": self.test_load,
            }
        )
        self.result_df = self.composition_test.join(result_df)

        if verbose and not skip_mae:
            print("MAE: ", mean_absolute_error(y_test, self.y_pred))
            print("R2: ", r2_score(y_test, self.y_pred))

        Path("results").mkdir(exist_ok=True)
        self.result_df.to_csv(join("results", "predicted_hv.csv"), index=False)
        print(
            "A file named predicted_hv.csv has been generated.\nPlease check your folder."
        )

        return self.y_pred

