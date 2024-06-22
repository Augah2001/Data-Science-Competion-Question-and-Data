from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import xgboost as xgb

model =     RandomForestClassifier(n_jobs=-1)

class Predict_Marital:
   

    def __init__(self, data, model=xgb.XGBClassifier()):
        self.predict_values = data.loc[data.marital_status.isna(),['gender','country',
        'location', 'loan_amount', 'number_of_defaults',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.features = data.loc[~data.marital_status.isna(),['gender','country',
        'location', 'loan_amount', 'number_of_defaults',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.target = data.loc[~data.marital_status.isna(),'marital_status']
        self.model = model

    def preprocess(self):
    
        one_hot = OneHotEncoder()
        le = LabelEncoder()

        transformer = ColumnTransformer([('one_hot', one_hot, ['gender', 'country',  'location'])],
                                        remainder='passthrough')
        self.features= transformer.fit_transform(self.features)
        self.target = self.target.map({'single':0, 'married':1, 'divorced': 2})
        
        self.predict_values = transformer.fit_transform(self.predict_values)
        
        return self.features, self.target
    

    def train(self):
        features, target = self.preprocess()
        
        self.model.fit(features, target)

        
        return self.model
    
    def predict(self):
        
        preds = self.model.predict(self.predict_values)
        return preds

    
class Predict_Job:
   

    def __init__(self, data, model=LassoCV()):
     
        self.predict_values = data.loc[data.job.isna(),['gender','location',
         'loan_amount', 'number_of_defaults','marital_status',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.features = data.loc[~data.job.isna(),['gender','marital_status','location',
         'loan_amount', 'number_of_defaults',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.target = data.loc[~data.job.isna(),'job']
        self.model = model
        print(self.predict_values.columns, self.features.columns)

    def preprocess(self):
        one_hot = OneHotEncoder()


        transformer = ColumnTransformer([('one_hot', one_hot, ['gender', 'marital_status', 'location', "location" ])],
                                        remainder='passthrough')
        transformer.fit(self.features)
        self.features= transformer.transform(self.features)
        self.target = self.target.map({'Teacher':0, 'Nurse':1, 'Doctor':2, 'Data Analyst':3, 'Software Developer':3,
       'Accountant':4, 'Lawyer':5, 'Engineer':6,'Data Scientist':7})
       
        self.predict_values = transformer.transform(self.predict_values)
        print(self.predict_values.shape, self.features.shape)
        
        return self.features, self.target
    

    def train(self):

        
        features, target = self.preprocess()
        X_train, X_test, y_train,y_test = train_test_split(features,target, test_size=0.2)
        
        
        self.model.fit(X_train, y_train)
        print(self.model.score(X_test, y_test))

        
        return self.model
    
    def predict(self):
        
        preds = self.model.predict(self.predict_values)
        return preds

    
    
 


