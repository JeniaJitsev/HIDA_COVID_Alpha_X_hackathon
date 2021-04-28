from pathlib import Path
from metrics import imputation_error_score, COLS
import pandas as pd
import numpy as np


class Tests():

   def __init__(self, input_path=None, output_dir=None):
        # Required parameters
        self.rng = np.random.RandomState(12)
        self.input_path = input_path
        self.output_dir = output_dir
        self.train_data = pd.read_csv(str(self.input_path) + '/trainSet.txt')
        self.test_data = pd.read_csv(str(self.input_path) + '/testSet.txt')

   def test_imputation_error_score(self):
        df = self.train_data[COLS]
        
        missing = pd.isna(df).values
        df_true = df.copy()
        df_true[pd.isna(df)] = 1
    
        df_pred = df.copy()
        df_pred[pd.isna(df)] = 0
        score = imputation_error_score(df_true, df_pred, missing) 
   
   def run(self):
        self.test_imputation_error_score()

if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[1])

    imputer = Tests(input_path=base_path + '/data/',
                                    output_dir=base_path + '/results/')
    imputer.run()