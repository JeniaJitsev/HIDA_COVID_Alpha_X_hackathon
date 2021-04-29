import datawig
import pandas as pd
import os
import statistics
import numpy as np
from pathlib import Path

class impute_datawig():

    def __init__(self, input_path):
        data_train = pd.read_csv(str(input_path) + '/trainSet.txt')
        data_test = pd.read_csv(input_path + "/testSet.txt")

        #data_train = data_train.set_index("PatientID")
        #data_test = data_test.set_index("PatientID")

        self.data = pd.concat([data_train, data_test])

    def fit(self):
        imputed_data = pd.DataFrame()

        data_to_impute = self.data.copy()
        data_to_impute.pop("PatientID")
        data_to_impute.pop("ImageFile")
        data_to_impute.pop("Hospital")
        data_to_impute.pop("Prognosis")
        self.imputer = []
        self.output_columns = []
        for i in range(len(data_to_impute.columns)):
            print(i)
            input_columns = list(data_to_impute.columns)
            input_columns.pop(i)
            self.output_columns.append(data_to_impute.columns[i])
            
            self.imputer.append(datawig.SimpleImputer(
                input_columns=input_columns, # column(s) containing information about the column we want to impute
                output_column= self.output_columns[i], # the column we'd like to impute values for
                ))
            self.imputer[i].fit(train_df=self.data, num_epochs=50)
    
    def predict(self, data_to_predict):
        data_to_impute = data_to_predict.copy()
        data_to_impute.pop("PatientID")
        data_to_impute.pop("ImageFile")
        data_to_impute.pop("Hospital")
        data_to_impute.pop("Prognosis")

        imputed_data = pd.DataFrame()
        categoricals = ["Sex", "Cough", "DifficultyInBreathing", "RespiratoryFailure", "CardiovascularDisease"]

        for i in range(len(self.imputer)):
            # output_columns = data_to_impute.columns[i]
            imputed = self.imputer[i].predict(data_to_impute)
            if self.output_columns[i] in categoricals:
                imputed[imputed.columns[-1]] = imputed[imputed.columns[-1]].astype(int)
            imputed_data[self.output_columns[i]] = imputed[imputed.columns[-1]]

        return imputed_data

    def prepare_imputed_data(self, data_to_predict, imputed_data, filepath = None):
        new_data = pd.DataFrame()
        for i in range(data_to_predict.shape[1]):
            print(i)
            column = data_to_predict.columns[i]
            new_data_column = data_to_predict[column]
            columns_impute = imputed_data.columns
            if column in list(columns_impute):
                indexes_nan = np.argwhere(np.isnan(data_to_predict[column]))
                imputed_data_column = imputed_data[column]
                new_data_column[np.isnan(new_data_column)] = imputed_data_column[np.isnan(new_data_column)]
            else:
                imputed_data_column = data_to_predict[column]
                new_data_column = imputed_data_column
            new_data[column] = new_data_column

        if filepath is not None:
            new_data.to_csv(filepath, sep=",")

        return new_data