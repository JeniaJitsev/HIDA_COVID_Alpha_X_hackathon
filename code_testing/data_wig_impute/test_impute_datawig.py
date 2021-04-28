import datawig
import pandas as pd
# from google.colab import files
import io
import statistics
import numpy as np
from pathlib import Path


base_path = str(Path(__file__).resolve().parents[1])

input_path=base_path + '/data/'

# data = pd.read_csv(io.StringIO(uploaded["trainSet.txt"].decode("utf-8")), sep=",")
# data = pd.read_csv(str(input_path) + '/trainSet.txt')

# data = pd.read_csv(str(input_path) + '/trainSet.txt', sep=",")
data = pd.read_csv(str(input_path) + '/trainSet.txt')

data = data.set_index("PatientID")
#data["Sex"] = data["Sex"].astype(str)
#data["Cough"] = data["Cough"].astype(str)
#data["DifficultyInBreathing"] = data["DifficultyInBreathing"].astype(str)
#data["RespiratoryFailure"] = data["RespiratoryFailure"].astype(str)
#data["CardiovascularDisease"] = data["CardiovascularDisease"].astype(str)

print(data)

data.pop("ImageFile")
data.pop("Hospital")
data.pop("Prognosis")

data_stdiv = data.std()

print(data_stdiv)

df_train, df_test = datawig.utils.random_split(data)

def score(data_in):
  names = data_in.columns
  imputed_data = data_in[names[-1]]
  gt_name = names[-1][0:-8]
  ground_truth = data_in[gt_name]
  score = (((imputed_data-ground_truth)/ data_stdiv[gt_name])**2)
  score_1 = []
  for i, value in enumerate(list(score)):
    if not np.isnan(value):
      score_1.append(min([value, 1]))

  score = statistics.mean(score_1)

  return score

score_all = []
imputed_data = pd.DataFrame()
categoricals = ["Sex", "Cough", "DifficultyInBreathing", "RespiratoryFailure", "CardiovascularDisease"]

for i in range(len(data.columns)):
  print(i)
  input_columns = list(data.columns)
  input_columns.pop(i)
  output_columns = data.columns[i]
  
  imputer = datawig.SimpleImputer(
      input_columns=input_columns, # column(s) containing information about the column we want to impute
      output_column= output_columns, # the column we'd like to impute values for
      )
  imputer.fit(train_df=df_train, num_epochs=50)
  imputed = imputer.predict(df_test)
  imputed_train = imputer.predict(data)
  if output_columns in categoricals:
    imputed[imputed.columns[-1]] = imputed[imputed.columns[-1]].astype(int)
    imputed_train[imputed_train.columns[-1]] = imputed_train[imputed_train.columns[-1]].astype(int)
  imputed_data[output_columns] = imputed_train[imputed_train.columns[-1]]
  score_all.append(score(imputed))

    

new_data = pd.DataFrame()
for i in range(data.shape[1]):
  column = data.columns[i]
  indexes_nan = np.argwhere(np.isnan(data[column]))
  new_data_column = data[column]
  imputed_data_column = imputed_data[column]
  new_data_column[np.isnan(new_data_column)] = imputed_data_column[np.isnan(new_data_column)]
  new_data[column] = new_data_column
    

print(new_data)

