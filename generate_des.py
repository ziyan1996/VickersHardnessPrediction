# -*- coding: utf-8 -*-
"""
@author: Ziyan Zhang, University of Houston
"""

# import general python package/ read in compounds list
import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
import matplotlib.pyplot as plt
from statistics import mean

df = pd.read_csv("hv_comp_load.csv")

df.head()
df.dtypes


class Vectorize_Formula:
    def __init__(self):
        elem_dict = pd.read_csv("elementsnew.csv")  # CHECK NAME OF FILE
        self.element_df = pd.DataFrame(elem_dict)
        self.element_df.set_index("Symbol", inplace=True)
        self.column_names = []
        for string in ["avg", "diff", "max", "min"]:
            for column_name in list(self.element_df.columns.values):
                self.column_names.append(string + "_" + column_name)

    def get_features(self, formula):
        try:
            fractional_composition = Composition(
                formula
            ).fractional_composition.as_dict()
            element_composition = Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            std_feature = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += (
                        self.element_df.loc[key].values * fractional_composition[key]
                    )
                    diff_feature = (
                        self.element_df.loc[list(fractional_composition.keys())].max()
                        - self.element_df.loc[list(fractional_composition.keys())].min()
                    )
                except Exception as e:
                    print(
                        "The element:",
                        key,
                        "from formula",
                        formula,
                        "is not currently supported in our database",
                    )
                    return np.array([np.nan] * len(self.element_df.iloc[0]) * 4)
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            std_feature = self.element_df.loc[list(fractional_composition.keys())].std(
                ddof=0
            )

            features = pd.DataFrame(
                np.concatenate(
                    [
                        avg_feature,
                        diff_feature,
                        np.array(max_feature),
                        np.array(min_feature),
                    ]
                )
            )
            features = np.concatenate(
                [
                    avg_feature,
                    diff_feature,
                    np.array(max_feature),
                    np.array(min_feature),
                ]
            )
            return features.transpose()
        except:
            print(
                "There was an error with the Formula: "
                + formula
                + ", this is a general exception with an unkown error"
            )
            return [np.nan] * len(self.element_df.iloc[0]) * 4


gf = Vectorize_Formula()

# empty list for storage of features
features = []

# add values to list using for loop
for formula in df["composition"]:
    features.append(gf.get_features(formula))

# feature vectors and targets as X and y
X = pd.DataFrame(features, columns=gf.column_names)
composition_load = pd.read_csv("hv_comp_load.csv")[["composition", "load"]]
predicted = composition_load.join(X)
predicted.to_csv("hv_des.csv", index=False)
print("A file named hv_des.csv has been generated.\nPlease check your folder.")
