import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from evaluation import COLS, full_evaluation, Submission

LABEL = "Prognosis"

class TabularBaseline(Submission):

    def fit(self, df_train):
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.clf = RandomForestClassifier()
        x = df_train[COLS].values
        self.imp.fit(x)
        x = self.imp.transform(x)
        y = df_train[LABEL].values
        self.clf.fit(x,y)

    def predict(self, df_test):
        df_test = df_test.copy()
        x = df_test[COLS].values
        x = self.imp.transform(x)
        y_pred = self.clf.predict(x)
        df_test[LABEL] = y_pred

        dummy = np.zeros_like(x)
        x[:] = np.nan
        x = self.imp.transform(x)
        df_test[COLS] = x
        return df_test

if __name__ == "__main__":
    full_evaluation(TabularBaseline)
