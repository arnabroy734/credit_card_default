import pandas as pd
from path.path import PREPROCESSED_DATA, BEST_MODEL
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from imblearn.pipeline import make_pipeline
from logs.logging import LOGGER
import logging
from xgboost import XGBClassifier
import pickle


def find_best_model():
    """
    1. Read data
    2. Split in train and test
    3. Fit data to each type of model
    4. Check the test f1 score for each model
    5. Save the best model with highest f1 score
    """
    best_model = None
    models = {
        'DT': DT(),
        'Logistic': Logistic(),
        'RF': RF(),
        'xgboost': XGB(),
        'SVM': SVM()

    }
    data = pd.read_csv(PREPROCESSED_DATA)
    X = data.drop(['default'], axis=1)
    y = data['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)
    cv = KFold(n_splits=5, random_state=400, shuffle=True)
    for model in models.keys():
        models[model].tune_hyperparameter(X_train, X_test, y_train, y_test, cv)
        if best_model is None:
            best_model = models[model]
        if models[model].test_f1 > best_model.test_f1:
            best_model = models[model]
    with open(BEST_MODEL, 'wb') as f:
        pickle.dump(best_model, f)
        f.close()
    LOGGER.log_training(message=f'Best model is {best_model}', level=logging.INFO)
    LOGGER.log_training(message=f"Best model test f1 - {best_model.test_f1}", level=logging.INFO)
    LOGGER.log_training(message=f"Best model test precision - {best_model.test_precision}", level=logging.INFO)
    LOGGER.log_training(message=f"Best model test recall - {best_model.test_recall}", level=logging.INFO)
    LOGGER.log_training(message=f"Best model test auc - {best_model.test_auc}\n\n", level=logging.INFO)


class DT:
    """
    Model - Simple decision tree with parameter tuning
    """

    def tune_hyperparameter(self, X_train, X_test, y_train, y_test, cv):
        LOGGER.log_training(message="Starting of training decision tree", level=logging.INFO)

        params = dict(
            splitter=['best', 'random'],
            max_depth=range(10, 50, 5),
        )

        grid_params = {'decisiontreeclassifier__' + key: params[key] for key in params}

        imb_pipeline = make_pipeline(
            # SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42),
            SMOTEENN(random_state=42),
            DecisionTreeClassifier(random_state=100)
        )

        grid_imb = GridSearchCV(imb_pipeline, param_grid=grid_params, scoring='f1', verbose=3, cv=cv,
                                return_train_score=True)
        grid_imb.fit(X_train, y_train)

        self.model = grid_imb.best_estimator_
        y_pred = self.predict(X_test)

        self.test_f1 = f1_score(y_test, y_pred)
        self.test_precision = precision_score(y_test, y_pred)
        self.test_recall = recall_score(y_test, y_pred)
        self.test_auc = roc_auc_score(y_test, y_pred)

        LOGGER.log_training(message=f"Test f1 - {self.test_f1}", level=logging.INFO)
        LOGGER.log_training(message=f"Test precision - {self.test_precision}", level=logging.INFO)
        LOGGER.log_training(message=f"Test recall - {self.test_recall}", level=logging.INFO)
        LOGGER.log_training(message=f"Test auc - {self.test_auc}", level=logging.INFO)
        LOGGER.log_training(message=f"Best model - {self.model}\n\n", level=logging.INFO)

    def predict(self, X):
        return self.model.predict(X)


class SVM:
    """
    Model - Support Vector Machine classifier
    """

    def tune_hyperparameter(self, X_train, X_test, y_train, y_test, cv):
        params = dict(
            C=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
        )
        grid_params = {'svc__' + key: params[key] for key in params}

        imb_pipeline = make_pipeline(
            SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42),
            # SMOTEENN(random_state=42),
            StandardScaler(),
            SVC(random_state=100)
        )

        grid_imb = GridSearchCV(imb_pipeline, param_grid=grid_params, scoring='f1', verbose=3, cv=cv,
                                return_train_score=True)
        grid_imb.fit(X_train, y_train)

        self.model = grid_imb.best_estimator_
        y_pred = self.predict(X_test)

        self.test_f1 = f1_score(y_test, y_pred)
        self.test_precision = precision_score(y_test, y_pred)
        self.test_recall = recall_score(y_test, y_pred)
        self.test_auc = roc_auc_score(y_test, y_pred)

        LOGGER.log_training(message=f"Test f1 - {self.test_f1}", level=logging.INFO)
        LOGGER.log_training(message=f"Test precision - {self.test_precision}", level=logging.INFO)
        LOGGER.log_training(message=f"Test recall - {self.test_recall}", level=logging.INFO)
        LOGGER.log_training(message=f"Test auc - {self.test_auc}", level=logging.INFO)
        LOGGER.log_training(message=f"Best model - {self.model}\n\n", level=logging.INFO)

    def predict(self, X):
        return self.model.predict(X)


class Logistic:
    """
    Model - Logistic Regression classifier
    """

    def tune_hyperparameter(self, X_train, X_test, y_train, y_test, cv):
        LOGGER.log_training(message="Starting of training logistic regression", level=logging.INFO)

        params = dict(
            C=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        )
        imb_pipeline = make_pipeline(
            SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42),
            # SMOTETomek( random_state=42),
            # SMOTEENN(random_state=42),
            StandardScaler(),
            LogisticRegression(random_state=100)
        )
        grid_params = {'logisticregression__' + key: params[key] for key in params}
        grid_imb = GridSearchCV(imb_pipeline, param_grid=grid_params, scoring='f1',
                                verbose=3, cv=cv, return_train_score=True)
        grid_imb.fit(X_train, y_train)

        self.model = grid_imb.best_estimator_
        y_pred = self.predict(X_test)

        self.test_f1 = f1_score(y_test, y_pred)
        self.test_precision = precision_score(y_test, y_pred)
        self.test_recall = recall_score(y_test, y_pred)
        self.test_auc = roc_auc_score(y_test, y_pred)

        LOGGER.log_training(message=f"Test f1 - {self.test_f1}", level=logging.INFO)
        LOGGER.log_training(message=f"Test precision - {self.test_precision}", level=logging.INFO)
        LOGGER.log_training(message=f"Test recall - {self.test_recall}", level=logging.INFO)
        LOGGER.log_training(message=f"Test auc - {self.test_auc}", level=logging.INFO)
        LOGGER.log_training(message=f"Best model - {self.model}\n\n", level=logging.INFO)

    def predict(self, X):
        return self.model.predict(X)


class RF:
    """
    Model - Random Forest classifier model
    """

    def tune_hyperparameter(self, X_train, X_test, y_train, y_test, cv):
        LOGGER.log_training(message="Starting of training random forest", level=logging.INFO)

        params = dict(
            n_estimators=[100, 150, 200],
            max_features=[0.6, 0.7, 0.8],
            max_samples=[0.1, 0.2, 0.3]
        )
        imb_pipeline = make_pipeline(
            # SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42),
            # SMOTETomek( random_state=42),
            SMOTEENN(),
            RandomForestClassifier(random_state=100, n_jobs=-1)
        )
        grid_params = {'randomforestclassifier__' + key: params[key] for key in params}
        grid_imb = GridSearchCV(imb_pipeline, param_grid=grid_params, scoring='f1',
                                verbose=3, cv=cv, return_train_score=True)
        grid_imb.fit(X_train, y_train)

        self.model = grid_imb.best_estimator_
        y_pred = self.predict(X_test)

        self.test_f1 = f1_score(y_test, y_pred)
        self.test_precision = precision_score(y_test, y_pred)
        self.test_recall = recall_score(y_test, y_pred)
        self.test_auc = roc_auc_score(y_test, y_pred)

        LOGGER.log_training(message=f"Test f1 - {self.test_f1}", level=logging.INFO)
        LOGGER.log_training(message=f"Test precision - {self.test_precision}", level=logging.INFO)
        LOGGER.log_training(message=f"Test recall - {self.test_recall}", level=logging.INFO)
        LOGGER.log_training(message=f"Test auc - {self.test_auc}", level=logging.INFO)
        LOGGER.log_training(message=f"Best model - {self.model}\n\n", level=logging.INFO)

    def predict(self, X):
        return self.model.predict(X)


class XGB:
    """
    Model - XGBClassifier model
    """

    def tune_hyperparameter(self, X_train, X_test, y_train, y_test, cv):
        LOGGER.log_training(message="Starting of training xgboost model", level=logging.INFO)

        params = dict(
            n_estimators=[100, 150, 200],
            max_depth=[3, 5, 7, 10],
            reg_lambda=[1, 2, 3]
        )
        imb_pipeline = make_pipeline(
            # SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42),
            SMOTEENN(random_state=42),
            XGBClassifier(random_state=100, n_jobs=-1, objective='binary:logistic')
        )
        grid_params = {'xgbclassifier__' + key: params[key] for key in params}
        grid_imb = GridSearchCV(imb_pipeline, param_grid=grid_params, scoring='f1',
                                verbose=3, cv=cv, return_train_score=True)
        grid_imb.fit(X_train, y_train)

        self.model = grid_imb.best_estimator_
        y_pred = self.predict(X_test)

        self.test_f1 = f1_score(y_test, y_pred)
        self.test_precision = precision_score(y_test, y_pred)
        self.test_recall = recall_score(y_test, y_pred)
        self.test_auc = roc_auc_score(y_test, y_pred)

        LOGGER.log_training(message=f"Test f1 - {self.test_f1}", level=logging.INFO)
        LOGGER.log_training(message=f"Test precision - {self.test_precision}", level=logging.INFO)
        LOGGER.log_training(message=f"Test recall - {self.test_recall}", level=logging.INFO)
        LOGGER.log_training(message=f"Test auc - {self.test_auc}", level=logging.INFO)
        LOGGER.log_training(message=f"Best model - {self.model}\n\n", level=logging.INFO)

    def predict(self, X):
        return self.model.predict(X)
