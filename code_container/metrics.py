import numpy as np
# from the table
# they use sigma=1 for boolean variables
COL_SIGMAS = [
    ("Age", 15.05),
    ("Temp_C", 0.97),
    ("WBC", 3.53),
    ("CRP", 66.93),
    ("Fibrinogen", 158.36),
    ("LDH", 235.22),
    ("Ddimer", 6743.13),
    ("Ox_percentage", 7),
    ("PaO2", 26.11),
    ("SaO2", 8.24),
    ("pH", 0.06),
    ("Sex", 1),
    ("Cough", 1),
    ("DifficultyInBreathing", 1),
    ("CardiovascularDisease", 1),
    ("RespiratoryFailure", 1),
]
COLS = [col for col, sigma in COL_SIGMAS]
SIGMAS = np.array([sigma for col, sigma in COL_SIGMAS])

def imputation_error_score(df_true, df_pred, missing):
    """
    X_true: groundtruth data matrix
        numpy array of shape (nb_examples, nb_features)
    X_pred: predicted matrix
        numpy array of shape (nb_examples, nb_features), same shape as X_true.
    missing: missing values boolean mask
        boolean numpy array of shape (nb_examples, nb_features), same shape as X_true
    """
    # Make sure that the dataframes follow the order in the variable COLS
    assert list(df_true.columns) == COLS
    assert list(df_pred.columns) == COLS
    X_true = df_true.values
    X_pred = df_pred.values
    assert (X_true.shape == X_pred.shape == missing.shape)
    # 1) compute squared error, normalize by std of each feature
    # 2) clip to the squared error to 1, to avoid outlier values
    m1 = np.minimum(((X_true - X_pred) / SIGMAS) ** 2, 1)
    # average only over missing entries (the ones we should predict)
    m1, missing = m1.flatten(), missing.flatten()
    return m1[missing].mean()
