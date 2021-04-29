import numpy as np
import pandas as pd
from evaluation import COLS

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from hebo.sklearn_tuner import sklearn_tuner

"""
As a final submission, we used :

    - task1. datawig imputation (improved over MICE)
    - task2. gradient boosted trees using imputed values of MICE (worked better than datawig) as input
"""

cls = GradientBoostingClassifier
# for RF
space_cfg = [
    {'name' : 'max_depth', 'type' : 'int', 'lb' : 1, 'ub' : 20},
    {'name' : 'min_samples_leaf', 'type' : 'num', 'lb' : 1e-4, 'ub' : 0.5},
    {'name' : 'max_features', 'type' : 'cat', 'categories' : ['auto', 'sqrt', 'log2']},
    # {'name' : 'bootstrap', 'type' : 'bool'},
    {'name' : 'min_impurity_decrease', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1.0},
    {'name' : 'learning_rate', 'type': 'num', 'lb': 1e-5, 'ub': 1},
]

df_train = pd.read_csv("../MICE_Imputation/trainSet_Imputed.txt")
# df_train = pd.read_csv("../data/trainSet_imputed.txt")

X_train = df_train[COLS].values
Y_train = df_train.Prognosis.values

IT = 50
result = sklearn_tuner(cls, space_cfg, X_train, Y_train, metric = accuracy_score, max_iter = IT)
print(result)

df_test = pd.read_csv("../MICE_Imputation/testSet_Imputed.txt")
# df_test = pd.read_csv("../data/testSet_imputed.txt")
X_test = df_test[COLS].values

clf = cls(**result)
clf.fit(X_train, Y_train)
y_pred_test = clf.predict(X_test)
df_test = pd.read_csv("../data/testSet_imputed.txt")
df_test["Prognosis"] = y_pred_test
assert np.all(~pd.isna(y_pred_test))
df_test.to_csv("COVID_Alpha_X.txt", index=False)
