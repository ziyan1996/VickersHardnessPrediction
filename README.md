# VickersHardnessPrediction
Predicting load dependent Vickers hardness based on chemical composition

## Citations

To cite the prediction of load dependent Vickers hardness, please reference the following work:

Zhang. Z, Tehrani. A.M., Oliynyk. A.O., Day. B and Brgoch. J, Finding Superhard Materials through Ensemble Learning, (add "submitted" after submission)

##  Prerequisites

To use the script provided here requires:

- [pymatgen](http://pymatgen.org)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/#)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [NumPy](https://docs.scipy.org/doc/numpy/index.html)
- [xlrd](https://xlrd.readthedocs.io/en/latest/index.html)

## Usage

*IMPORTANT* To use all scripts smoothly, please clone the entire repo with all files in one folder and work within that folder.

To train the model and predict the hardness of some materials you are interested in, simply following these steps:

### 1 Generate descriptors

Firstly, prepare your compositions in an excel file, and name it `pred_hv_comp.xlsx` so that the script can recognize this file. The first column of the `pred_hv_comp.xlsx` file should be named as `Composition`.

To generate descriptors for your compositions, simply run:

```bash
python generate_des.py
```

You will have an output file named `pred_hv_descriptors.xlsx` containing all compositional descriptors.

*IMPORTANT STEP:* now please manualy add a new column with load values (unit: N) at the end of the descriptor file you just generated. It is up to you at which load you want to predict the hardness.

### 2 Train the model and make prediction of your compounds

We have provided the training dataset in the file `hv_comp_load.xlsx` where you will find chemical compositions, hardness value and corresponding load value. We also provided the descriptors of our training set (`hv_des.xlsx`). The training process of our model will be automatically done when you run the prediciton script as this:

```bash
python hv_prediction.py
```

Results will be stored in a file named `predicted_hv.xlsx`. Basically the script will first train the model using the dataset we constructed, then read the `pred_hv_descriptors.xlsx` file you just generated and give you the predicted hardness at any load value you would be interested in.

## Authors

This code was created by [Ziyan Zhang](https://github.com/ziyan1996) who is advised by [Jakoah Brgoch](https://www.brgochchemistry.com/).
