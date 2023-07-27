from path.path import UPLOADED_DATA, PREPROCESSED_DATA, PREDICTION_INPUT, PREDICTION_PREPROCESSED, PREPROCESSOR
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from logs.logging import LOGGER
import logging
import pprint
import pickle

class Preprocessor:
    def __init__(self):
        self.pipeline = Pipeline([
            ('replace', Replace()),
            ('encoding', Encoding())
        ])

    def train_preprocess(self):

        """
        Description:
        1. Read uploaded data
        2. Preprocess data using pipeline
        """
        try:
            LOGGER.log_preprocessing(message="Start of preprocessing training data", level=logging.INFO)
            data = pd.read_excel(UPLOADED_DATA, index_col='ID')
            X = data.drop(['default'], axis=1)
            y = data['default']
            X = self.pipeline.fit_transform(X, y)
            X['default'] = y
            X.to_csv(PREPROCESSED_DATA, index=False)
            # Save preprocessing file
            with open(PREPROCESSOR, 'wb') as f:
                pickle.dump(self, f)
                f.close()
            LOGGER.log_preprocessing(message=f"Successful end of preprocessing. Preprocessor saved as {PREPROCESSOR}\n\n", level=logging.INFO)
        except Exception as e:
            LOGGER.log_preprocessing(message=f"Error in preprocessing {e}\n\n", level=logging.ERROR)
    
    def prediction_preprocess(self):
        """
        Description:
        1. Read uploaded data
        2. Preprocess data using pipeline
        3. Return (True, data) if preprocessing is successful
        4. Return (False, None) if preprocessing goes wrong 
        """
        try:
            LOGGER.log_preprocessing(message="Start of preprocessing predcition data", level=logging.INFO)
            X = pd.read_excel(PREDICTION_INPUT, index_col='ID')
            X = self.pipeline.transform(X)
            X.to_csv(PREDICTION_PREPROCESSED, index=False)
            LOGGER.log_preprocessing(message="Successful end of preprocessing of prediction data \n\n", level=logging.INFO)
            return (True, X)
        except Exception as e:
            LOGGER.log_preprocessing(message=f"Error in preprocessing prediction data {e}\n\n", level=logging.ERROR)
            return (False, None)



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
    
    def transform(self, X, y=None):
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

    def give_feature_encoding(self, data, feature, target):
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

    
        return fearure_encoding


    def fit(self, X, y):
        data = X.copy()
        data['default'] = y

        self.encodings = {}
        for feature in ['SEX', 'EDUCATION', 'MARRIAGE']:
            self.encodings[feature] = self.give_feature_encoding(data, feature, 'default')
        
        for i in range(1,7):
            feature = 'PAY_' + str(i)
            self.encodings[feature] = self.give_feature_encoding(data, feature, 'default')
        
        return self
    
    def transform(self, X, y=None):
        for feature in self.encodings.keys():
            feature_encoding = self.encodings[feature]
            X[feature] = X[feature].map(lambda x: feature_encoding[x])
        return X