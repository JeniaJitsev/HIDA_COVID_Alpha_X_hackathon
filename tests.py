from metrics import imputation_error_score, COLS
import pandas as pd
import numpy as np

def test_imputation_error_score():
    df = pd.read_csv("trainSet.txt")
    df = df[COLS]
    
    missing = pd.isna(df).values
    df_true = df.copy()
    df_true[pd.isna(df)] = 1

    df_pred = df.copy()
    df_pred[pd.isna(df)] = 0
    score = imputation_error_score(df_true, df_pred, missing) 
