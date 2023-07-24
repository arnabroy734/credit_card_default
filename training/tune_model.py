import pandas as pd
from path.path import PREPROCESSED_DATA
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def find_best_model():
    """
    1. Read data
    2. Split in train and test
    3. Upsample train data with SMOTE
    """
    models = {
        # 'DT' : DT(), 
        'SVM' : SVM(), 
        # 'Logistic' : Logistic()
        # 'RF' : RF(),
        # 'Adaboost' : AdaBoost()
    }
    data = pd.read_csv(PREPROCESSED_DATA)
    X = data.drop(['default'], axis=1)
    y = data['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 400)
    sampler = SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42)
    X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train)

    for model in models.keys():
        models[model].tune_hyperparameter(X_train_sampled, X_test, y_train_sampled, y_test)




class DT:
    """
    Model - Simple decision tree with parameter tuning
    """
    def tune_hyperparameter(self, X_train, X_test, y_train, y_test):
        params = dict(
            splitter=['best', 'random'], 
            max_depth = range(10, 50, 5),
            
        )
        clf = GridSearchCV(DecisionTreeClassifier(random_state=100), param_grid=params,scoring='f1', verbose=3)
        clf.fit(X_train, y_train)
        self.model = clf.best_estimator_
        y_pred = self.model.predict(X_test)
        print("Test f1 - ", f1_score(y_test, y_pred))
        print("Test precision - ", precision_score(y_test, y_pred))
        print("Test recall  - ", recall_score(y_test, y_pred))
        print("Test AUC - ", roc_auc_score(y_test, y_pred))

class SVM:
    """
    Model - Support Vector Machine classifier
    """
    def tune_hyperparameter(self, X_train, X_test, y_train, y_test):
        params = dict(
            C=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
            # kernel = ['rbf', 'poly', 'sigmoid']
            
        )
        clf = GridSearchCV(SVC(random_state=100), param_grid=params,scoring='f1', verbose=3)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        clf.fit(X_train_scaled, y_train)
        self.model = clf.best_estimator_
        y_pred = self.model.predict(self.scaler.transform(X_test))
        print("Test f1 - ", f1_score(y_test, y_pred))
        print("Test precision - ", precision_score(y_test, y_pred))
        print("Test recall  - ", recall_score(y_test, y_pred))
        print("Test AUC - ", roc_auc_score(y_test, y_pred))
        
        
class Logistic:
    """
    Model - Logistic Regression classifier
    """
    def tune_hyperparameter(self, X_train, X_test, y_train, y_test):
        params = dict(
            C=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        )
        clf = GridSearchCV(LogisticRegression(random_state=100), param_grid=params,scoring='f1', verbose=3)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        clf.fit(X_train_scaled, y_train)
        self.model = clf.best_estimator_
        y_pred = self.model.predict(self.scaler.transform(X_test))
        print("Test f1 - ", f1_score(y_test, y_pred))
        print("Test precision - ", precision_score(y_test, y_pred))
        print("Test recall  - ", recall_score(y_test, y_pred))
        print("Test AUC - ", roc_auc_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

class RF:
    """
    Model - Random Forest classifier model
    """
    def tune_hyperparameter(self, X_train, X_test, y_train, y_test):
        params = dict(
            # n_estimators = [100, 150, 200],
            # max_features = [0.6, 0.7, 0.8, 0.9, 1],
            max_samples = [0.1, 0.2, 0.3, 0.4, 0.5]
        )
        clf = GridSearchCV(RandomForestClassifier(random_state=100, n_jobs=-1), param_grid=params,scoring='f1', verbose=3)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        clf.fit(X_train_scaled, y_train)
        self.model = clf.best_estimator_
        y_pred = self.model.predict(self.scaler.transform(X_test))
        print("Test f1 - ", f1_score(y_test, y_pred))
        print("Test precision - ", precision_score(y_test, y_pred))
        print("Test recall  - ", recall_score(y_test, y_pred))
        print("Test AUC - ", roc_auc_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

class AdaBoost:
    """
    Model - Adaboost Classifier
    """
    def tune_hyperparameter(self, X_train, X_test, y_train, y_test):
        params = dict(
            # n_estimators = [50, 60, 80, 100],
            learning_rate = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        )
        clf = GridSearchCV(AdaBoostClassifier(random_state=100), param_grid=params,scoring='f1', verbose=3)
        clf.fit(X_train, y_train)
        self.model = clf.best_estimator_
        y_pred = self.model.predict(X_test)
        print("Test f1 - ", f1_score(y_test, y_pred))
        print("Test precision - ", precision_score(y_test, y_pred))
        print("Test recall  - ", recall_score(y_test, y_pred))
        print("Test AUC - ", roc_auc_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))