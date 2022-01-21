import numpy as np


def log_cosh_quantile(alpha):
    """Log cosh quantile is a regularized quantile loss function.
    Source: # https://towardsdatascience.com/confidence-intervals-for-xgboost-cac2955a8fde
    Parameters
    ----------
    alpha : float
        confidence level (e.g. 95% confidence == 0.95). Ranges between 0.0 and 1.0.
    """

    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err) ** 2
        return grad, hess

    return _log_cosh_quantile
