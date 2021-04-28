"""
Simple evaluation code that can be used
to evaluate a submission using cross validation.

We need a common codebase for  evaluating different
methods. Since the data is small, we can afford
to use cross validation to get an estimate on
the performance with an uncertainty estimate.

each submission is a class following scikit-learn API.
It has a `fit` and `predict` method.
`fit` is used for training phase, `predict` for testing phase.
the same model is evaluated several times using cross validation.

"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from metrics import imputation_error_score, COLS

# cross validation scheme
N_SPLITS = 10
N_REPEATS = 1
RANDOM_STATE = 10101

# Abstract submission class
class Submission:

    def fit(self, df_train):
        pass

    def predict(df_test):
        """
        return a dataframe (nb_examples, nb_features) which identical shape as df_test.

        task1. for each row, for each col in `COLS`, we predict the value of col given the other `COLS`.
        In the end we get a complete imputed dataframe. Please make sure all values are imputed
        whether they are Nans or not. That is, we need to predict all the nb_examples x nb_features
        because it is needed for evaluation.

        task2. replace the col `prognosis` values by either SEVERE or MILD

        """
        raise NotImplementedError()


class DummySubmission(Submission):
    """
    simple dummy submission for an example
    """
    
    def predict(self, df_test):
        # predict SEVERE for everything
        df_test["Prognosis"] = "SEVERE"
        # replace missing data with 0
        df_test = df_test.fillna(0)
        df_test[COLS] = 0
        return df_test

def evaluate(submission_cls, input_path="."):
    
    df_train = pd.read_csv(str(input_path) + '/trainSet.txt')
    df_test = pd.read_csv(str(input_path) + '/testSet.txt')

    random_state = RANDOM_STATE
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=random_state)
    metrics = []
    for train_index, valid_index in rkf.split(df_train):
        train = df_train.iloc[train_index]
        valid = df_train.iloc[valid_index]
        submission = submission_cls()
        submission.fit(train)

        valid_ = valid.copy()

        # hide labels
        valid_["Prognosis"] = np.nan

        pred_valid = submission.predict(valid_)
        #ensure no missing data is left
        assert np.all(~pd.isna(pred_valid))
        
        pred_prognosis = pred_valid["Prognosis"]
        true_prognosis = valid["Prognosis"]
        metrics.append({
            "prognosis_accuracy": (pred_prognosis==true_prognosis).mean(),
            "imputation_error": imputation_error_score(valid_[COLS], pred_valid[COLS], ~(pd.isna(valid_[COLS])).values )  ,
        })
    submission = submission_cls()
    submission.fit(df_train)
    pred_test = submission.predict(df_test)
    # pred_test[~pd.isna(df_test)] = 0#df_test[~pd.isna(df_test)]
    pred_test = pred_test.mask(~pd.isna(df_test), df_test)
    #ensure no missing data is left
    assert np.all(~pd.isna(pred_test))
    return metrics, pred_test 
    
def display_metrics(metrics):
   names = metrics[0].keys()
   for name in names:
       vals = np.array([m[name] for m in metrics])
       mean = vals.mean()
       std = vals.std()
       print(f"{name}: {mean} ± {std}")

if __name__ == "__main__":
    base_path = str(Path(__file__).resolve().parents[1])
    metrics, pred_test = evaluate(DummySubmission, input_path=base_path + '/data/')
    display_metrics(metrics)
    # final submission
    pred_test.to_csv("test_submission.csv", index=False)
