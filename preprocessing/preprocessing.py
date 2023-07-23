from path.path import UPLOADED_DATA
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from logs.logging import LOGGER
import logging

class Preprocessor:
    def __init__(self):
        self.pipeline = Pipeline([
            ('replace', Replace())
        ])

    def preprocess(self):

        """
        Description:
        1. Read uploaded data
        """
        try:
            data = pd.read_excel(UPLOADED_DATA, index_col='ID')
            X = data.drop(['default'], axis=1)
            y = data['default']
            X = self.pipeline.fit_transform(X)
            print(X.head(5))
        except Exception as e:
            LOGGER.log_preprocessing(message=f"Error in preprocessing {e}", level=logging.ERROR)


class Replace(BaseEstimator, TransformerMixin):
    """
    This class replaces some values on following features:

    Education:
    1. As per data description there are only categories 1, 2, 3 and 4 where 4 is for others
    2. In the dataset total 7 categories are present - so we will replace all categories (apart from 1, 2 and 3) as 4

    Marriage:
    1. As per data description there are only categories 1, 2 and 3 where 3 is for others
    2. But in dataset 0 is present as a separate category. So, replace 0 with 3

    PAY_X
    3. From EDA it is clear that for every PAY_X category there are very less number of records where PAY_X >= 3
    4. So, we will replace all PAY_X >=3 values with 3
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['EDUCATION'] = X['EDUCATION'].map(lambda x : 4 if x not in [1,2,3] else x)
        X['MARRIAGE'] = X['MARRIAGE'].map(lambda x: 3 if x not in [1,2] else x)
        for i in range(1,7):
            feature = 'PAY_' + str(i)
            X[feature] = X[feature].map(lambda x: 3 if x>=3 else x)
        return X

class Encoding(BaseEstimator, TransformerMixin):
    """
    This class encodes feature Marriage, Sex, Education and PAY_X by probability ratio
    """

    def give_feature_encoding(data, feature, target):
        """
        Description: Probability ratio encoding
        Parameters: data, feature, target
        Description: 
            1. Calculate prob (target=1 | feature = X)
            2. Calculate prob (target=0 | feature = X)
            3. feature_encoding[X] = prob (target=1 | feature = X) / prob (target=0 | feature = X)
        """
        feature_group = data.groupby([feature, target]).size()/data.groupby(feature)[target].count()
        fearure_encoding = {}
        for x in data[feature].unique():
            fearure_encoding[x] = feature_group[(x,1)]/feature_group[(x,0)]
            # fearure_encoding[x] = feature_group[(x,0)]

    
        return fearure_encoding
