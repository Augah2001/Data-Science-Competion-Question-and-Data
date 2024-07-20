from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

# ... (assuming you have your DataFrame 'df' loaded)




class Predict_Marital:
        def __init__(self,  model=None):
            if model is None:
                model = xgb.XGBClassifier()
            self.model = model

       

        def fit(self, X, y=None):

            
            features = X.loc[X['marital_status'] != ' '].drop(columns='marital_status')
            target = X.loc[X['marital_status'] != ' ', 'marital_status']
            target = target.map({'single': 0, 'married': 1, 'divorced': 2}).astype('int')

           
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=81, stratify=target,  )
            self.model.fit(X_train, y_train)
            print(f'Predict_Marital Model Score: {self.model.score(X_test, y_test)}')
            return self

        def transform(self,X):
            predict_values = X.loc[X['marital_status'] == ' '].drop(columns=['marital_status'])
            preds = self.model.predict(predict_values)

            X.loc[X['marital_status'] == ' ', 'marital_status'] = pd.DataFrame(preds)[0].map({0: 'single', 1: 'married', 2: 'divorced'}).to_numpy()
           
        # # Encode marital status
            X['marital_status'] = X['marital_status'].map({'single': 0, 'married': 1, 'divorced': 2}).astype('int')
          
            
            return X 


class DataTransformer(BaseEstimator, TransformerMixin):

    
    
    def __init__(self, features, labels, threshold = 0.2):
        self.label_encoders = {}
        self.features = features
        self.labels = labels
        self.location_loan_index_encode = [{13: 0, 12: 1, 3: 2, 4: 3, 18: 4, 21: 5, 20: 6, 6: 7, 15: 8, 24: 9, 16: 10, 22: 11, 8: 12, 1: 13, 9: 14, 7: 15, 14: 16, 5: 17, 0: 18, 2: 19, 10: 20, 19: 21, 17: 22, 23: 23, 11: 24}]
        self.job_loan_index_encode = [{4: 0, 7: 1, 3: 2, 6: 3, 0: 4, 1: 5, 5: 6, 8: 7, 2: 8}]
        self.location_ratios = None
        self.threshold = threshold
        self.isFitted = False
       
        

    def fit(self, X, y=None):
        categorical_features = [ 'job', 'location']

        for cat in categorical_features:
            le = LabelEncoder()
            le.fit(X[cat])
            self.label_encoders[cat] = le

        
        # X_features = self.transform(self.features.copy())[[ 'location']]
        # X_features['y'] = self.labels.map({'Did not default':0, 'Defaulted':1})
        # print(X_features)
        # self.location_ratios = X_features.groupby('location')['y'].mean()
        # self.location_ratios = self.location_ratios / (1 - self.location_ratios)
     
    
       


        
        
        
            

        return self

    def transform(self, X):
        
       
        X = X.copy()
        categorical_features = [ 'job', 'location']

        for cat in categorical_features:
            self.label_encoders[cat].classes_ = np.append(self.label_encoders[cat].classes_, '<unknown>')
            X[cat] = self.label_encoders[cat].transform(X[cat])
        
        X['gender'] = X['gender'].map({"other": 1, "female": 0, 'male': 2}).astype('int')
        
        
        
        
           
        X['location'] = X['location'].map(self.location_loan_index_encode[0])
        # print(X['job'])
        X['job'] = X['job'].map(self.job_loan_index_encode[0]).astype('int')
        

#         # independent binary features for loan_amount === 5000 and greater than 75000
        X['loan_equal_5000'] = np.where(X.loan_amount==5000,1,0)
        X['loan_>_75000'] = np.where(X.loan_amount>75000,1,0)
        X = X.assign(total_loan_amount_per_job=X.groupby('job')['loan_amount'].transform('mean'))

        ### Create new feature to capture job and location relationship
        X['job_location_interact'] = np.log1p(np.sqrt(X['job']))/(X['location']+1)

        # if self.isFitted == True:
        #     X['location_class'] = X['location'].map(lambda loc: 1 if self.location_ratios[loc] < self.threshold else 0)  
              
        # print(X.columns)
        X = self.create_new_features(X)
          
        return X
    
    def create_new_features(self,X):
        epsilon = 1e-8  # Small value to avoid division by zero
        X = X.copy()  # To avoid modifying the original DataFrame
        X['interest_rate_per_loan_amount'] = X['interest_rate'] / (X['total_loan_amount_per_job'] + epsilon)
        X['age_per_loan_amount'] = X['age'] / (X['total_loan_amount_per_job'] + epsilon)
        X['job_location_per_loan_amount'] = X['job_location_interact'] / (X['total_loan_amount_per_job'] + epsilon)
        X['loan_5000_and_loan_75000_interaction'] = X['loan_equal_5000'] * X['loan_>_75000']
       
        return X






