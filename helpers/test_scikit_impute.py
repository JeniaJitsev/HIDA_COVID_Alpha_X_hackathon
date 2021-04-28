# -- Testing fancyimpute in Sklearn

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


if __name__ == "__main__":
    imp_mean = IterativeImputer(random_state=0)
    
    D = [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]
    res = imp_mean.fit(D)

    print(res)

    X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    res = imp_mean.transform(X)

    print(res)

    
    