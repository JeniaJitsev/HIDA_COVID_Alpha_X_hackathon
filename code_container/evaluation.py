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
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, ShuffleSplit
from metrics import imputation_error_score, COLS

# cross validation scheme
N_SPLITS = 2
N_REPEATS = 1
RANDOM_STATE = 10101
MISSING_IMAGES_RATE = 31/210

# Abstract submission class
class Submission:
    
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def fit(self, df_train):
        pass

    def predict(df_test):
        """
        return a dataframe (nb_examples, nb_features) with identical shape as df_test.

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
        # replace the remaining cols with 0
        df_test[COLS] = 0
        return df_test

def evaluate(submission_cls, input_path="."):
    print(f"Evaluating '{submission_cls.__name__}'...")
    train_path = os.path.join(input_path, 'trainSet')
    test_path = os.path.join(input_path, 'testSet')

    df_train = pd.read_csv(os.path.join(train_path, 'trainSet.txt'))
    df_test = pd.read_csv(os.path.join(test_path, 'testSet.txt'))

    random_state = RANDOM_STATE
    rng = np.random.RandomState(random_state)
    # rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=random_state)
    rkf = ShuffleSplit(n_splits=2, random_state=random_state, train_size=0.85)
    metrics = []
    for train_index, valid_index in rkf.split(df_train):
        train = df_train.iloc[train_index]
        valid = df_train.iloc[valid_index]
        submission = submission_cls(train_path=train_path, test_path=train_path)
        submission.fit(train)

        valid_ = valid.copy()

        # hide labels
        valid_["Prognosis"] = np.nan
        # artifically make some images missing (like the actual test set)
        # missing_images = (rng.uniform(size=len(valid_)) <= MISSING_IMAGES_RATE)
        # valid_.loc[missing_images, "ImageFile"] = np.nan

        # prediction

        pred_valid = submission.predict(valid_)
        #ensure no missing data is left
        # assert np.all(~pd.isna(pred_valid))
        
        pred_prognosis = pred_valid["Prognosis"]
        true_prognosis = valid["Prognosis"]
        metrics.append({
            "prognosis_accuracy": (pred_prognosis==true_prognosis).mean(),
            # compute imputation error only on non-nan values (we don't know the groundtruth values
            # for the ones which are alreay Nan)
            "imputation_error": imputation_error_score(valid_[COLS], pred_valid[COLS], ~(pd.isna(valid[COLS])).values )  ,
        })
    submission = submission_cls(train_path=train_path, test_path=test_path)
    submission.fit(df_train)
    pred_test = submission.predict(df_test)
    # make sure the non-nan values in `df_test` are
    # identical to the ones in `pred_test`
    pred_test = pred_test.mask(~pd.isna(df_test), df_test)
    #ensure no missing data is left
    # assert np.all(~pd.isna(pred_test))

    return metrics, pred_test 
    
def display_metrics(metrics):
   names = metrics[0].keys()
   for name in names:
       vals = np.array([m[name] for m in metrics])
       mean = vals.mean()
       std = vals.std()
       print(f"{name}: {mean} ± {std}")

def full_evaluation(submission_cls):
    base_path = str(Path(__file__).resolve().parents[1])
    metrics, pred_test = evaluate(submission_cls, input_path=base_path + '/data/')
    display_metrics(metrics)
    pred_test.to_csv("test_submission.csv", index=False)

if __name__ == "__main__":
    full_evaluation(DummySubmission)
