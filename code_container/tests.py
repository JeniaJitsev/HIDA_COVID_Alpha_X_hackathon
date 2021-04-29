from pathlib import Path
from metrics import imputation_error_score, COLS
import pandas as pd
import numpy as np
import os

class Tests():

   def __init__(self, input_path=None, input_dir_missing=None, input_dir_imputed = None):
        # Required parameters
        self.rng = np.random.RandomState(12)
        self.input_path = input_path
        self.input_dir_missing = input_dir_missing
        self.input_dir_imputed= input_dir_imputed
        self.df_true = pd.read_csv(str(self.input_path) + '/trainSet.txt')
        self.df_true = self.df_true [COLS]
   
   def test_imputation_error_score(self):
        df = self.train_data[COLS]

        missing = pd.isna(df).values
        df_true = df.copy()
        df_true[pd.isna(df)] = 1

        df_pred = df.copy()
        df_pred[pd.isna(df)] = 0
        score = imputation_error_score(df_true, df_pred, missing)
        return score

   def run(self):        
        for file in os.listdir(self.input_dir_imputed):
            df_pred =  pd.read_csv(str(self.input_dir_imputed) + file)
            df_pred = df_pred[COLS]
            missing = pd.read_csv(str(self.input_dir_missing) + 'info_missing_'+file[13:])
            missing = missing[COLS].values  
            missing_everywhere = pd.isna(self.df_true).values #the ones that are also missing in original Set should not be uesed for computing score
            missing[missing_everywhere]=False                               
            score = imputation_error_score(self.df_true, df_pred,missing)
            print(f"{file}: {score}")
        return score

if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[1])

    imputer = Tests(input_path=base_path + '/data',
                                    input_dir_missing=base_path + '/data/missing_val_info/',
                                    input_dir_imputed= base_path+'/data/train_set_with_missing_vals/imputed_data_datawig/')
    score = []
    score.append(imputer.run())
    print('Score:', score)
