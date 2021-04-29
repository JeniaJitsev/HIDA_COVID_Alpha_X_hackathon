from pathlib import Path
from metrics import imputation_error_score, COLS
import pandas as pd
import numpy as np


class Generate_Test_Sets():

   def __init__(self, input_path=None, output_dir_test=None, output_dir_missing=None, output_dir_imputed = None):
        # Required parameters
        self.rng = np.random.RandomState(12)
        self.input_path = input_path
        self.output_dir_test = output_dir_test
        self.output_dir_missing = output_dir_missing
        self.output_dir_imputed = output_dir_imputed
        self.train_data = pd.read_csv(str(self.input_path) + '/trainSet.txt')

   def generate_missing_value_train_data(self, missing_rate = 0.30):
        """
            Generate Matrix from which a 30% of data is additionally missing
            randomly
        """
        n_samples, n_features = self.train_data.shape
        n_missing_samples = int(n_samples * missing_rate) #samples missing in each col        
        missing_samples = np.zeros(n_samples, dtype=bool)        
        missing_samples[: n_missing_samples] = True
        self.X_missing = self.train_data.copy().set_index("PatientID")
        self.missing_info = self.train_data.copy().set_index("PatientID")
        
        for c in COLS: 
            self.rng.shuffle(missing_samples)
            self.X_missing[c][missing_samples] = np.nan
            self.missing_info[c] = missing_samples
        
        self.dummy_imputed = self.X_missing.copy()
        for c in COLS: 
            self.dummy_imputed[c][self.dummy_imputed[c].isna()] = 0
            
            
   def run(self, n = 5, missing_rate= 0.3):
        for i in range(n):
            self.generate_missing_value_train_data(missing_rate = missing_rate)    
            self.X_missing.to_csv(self.output_dir_test+f"test_missing_{int(missing_rate*100)}perc_randomly_{i}.txt", na_rep='NaN')
            self.missing_info.to_csv(self.output_dir_missing+f"info_missing_{int(missing_rate*100)}perc_randomly_{i}.txt", na_rep='NaN')
            self.dummy_imputed.to_csv(self.output_dir_imputed+f"imputed_{int(missing_rate*100)}perc_randomly_{i}.txt", na_rep='NaN')            
            print(f"Generated new testSet with {missing_rate}% NaN per col.")
            
if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[1])

    generator = Generate_Test_Sets(input_path=base_path + '/data/',
                                    output_dir_test=base_path + '/data/train_set_with_missing_vals/',
                                    output_dir_missing= base_path+'/data/missing_val_info/',
                                    output_dir_imputed= base_path+'/data/imputed/')
    
    generator.run(n = 5, missing_rate= 0.3)
    generator.run(n = 5, missing_rate= 0.25)
    generator.run(n = 5, missing_rate= 0.2)
    generator.run(n = 5, missing_rate= 0.15)
    generator.run(n = 5, missing_rate= 0.1)
