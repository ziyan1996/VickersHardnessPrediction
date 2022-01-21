"""Test VickersHardness class."""
from os.path import join
import pandas as pd
from sklearn.model_selection import train_test_split

from vickers_hardness.vickers_hardness_ import VickersHardness


def test_vickers_hardness():
    # %% load dataset
    X = pd.read_csv(join("vickers_hardness", "data", "hv_des.csv"))
    prediction = pd.read_csv(join("vickers_hardness", "data", "hv_comp_load.csv"))
    y = prediction["hardness"]

    # %% Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9, test_size=0.1, random_state=100, shuffle=True
    )

    vh = VickersHardness(hyperopt=False)
    vh.fit(X_train, y_train)
    vh.predict(X_test, y_test)


if __name__ == "__main__":
    test_vickers_hardness()
