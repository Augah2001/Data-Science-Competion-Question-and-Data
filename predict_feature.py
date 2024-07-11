from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import xgboost as xgb
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.linear_model import LassoCV

#fitting random forest to my model 
clf = RandomForestClassifier()
clf = xgb.XGBClassifier()
# clf = CatBoostClassifier()

model =     RandomForestClassifier(n_jobs=-1)

class Predict_Marital:
   

    def __init__(self, data, model=clf):
        self.predict_values = data.loc[data['marital_status'] == ' '].drop(columns = ['marital_status'])
        self.features = data.loc[data['marital_status'] != ' '].drop(columns = 'marital_status')
        self.target = data.loc[data['marital_status'] != ' ','marital_status']
        self.model = model

    def preprocess(self):
    
        
        self.target = self.target.map({'single':0, 'married':1, 'divorced': 2}).astype('int')
        
        
        
        return self.features, self.target
    

    def train(self):
        features, target = self.preprocess()
        X_train, X_test, y_train,y_test = train_test_split(features,target, test_size=0.2)
        
        
        self.model.fit(X_train, y_train)
        print(self.model.score(X_test, y_test))

        
        return self.model
    
    def predict(self):

        
        preds = self.model.predict(self.predict_values)
        feature_importances = pd.DataFrame(self.model.feature_importances_).T
        feature_importances.columns = self.model.feature_names_in_
        feature_importances = feature_importances.T.sort_values(by=0)

        return preds, feature_importances
class Predict_outstanding:
   

    def __init__(self, data, model=clf):

    
        
        condition = (((data['loan_amount'])- (data['outstanding_balance'])) <0)
        self.predict_values = data.loc[condition, :].drop(columns = ['outstanding_balance',  'salary', 'remaining term', 'number_of_defaults', 'location'])

        self.features = data.loc[(((data['loan_amount'])- (data['outstanding_balance'])) >0), :].drop(columns = ['outstanding_balance',  'salary', 'remaining term', 'number_of_defaults', 'location'])


        self.target = data.loc[((data['loan_amount'])- (data['outstanding_balance'])) >0,'outstanding_balance']
        self.model = model


    def preprocess(self):
    
        
        
        
        
        return self.features, self.target
    

    def train(self):
        features, target = self.preprocess()
        X_train, X_test, y_train,y_test = train_test_split(features,target, test_size=0.2)
        
        
        self.model.fit(X_train, y_train)
        print(self.model.score(X_test, y_test))
        y_p = self.model.predict(X_test)

        
        return self.model, y_p, y_test
    
    def predict(self):

        
        preds = self.model.predict(self.predict_values)
        # feature_importances = pd.DataFrame(self.model.feature_importances_).T
        # feature_importances.columns = self.model.feature_names_in_
        # feature_importances = feature_importances.T.sort_values(by=0)

        return preds, 
    

    
class Predict_Job:
   

    def __init__(self, data, model=xgb.XGBClassifier()):
     
        self.predict_values = data.loc[data.location.isna(),['gender','job',
         'loan_amount', 'number_of_defaults','marital_status',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.features = data.loc[~data.location.isna(),['gender','marital_status','job',
         'loan_amount', 'number_of_defaults',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.target = data.loc[~data.location.isna(),'location']
        self.model = model
        print(self.predict_values.columns, self.features.columns)

    def preprocess(self):
        one_hot = OneHotEncoder()


        transformer = ColumnTransformer([('one_hot', one_hot, ['gender', 'marital_status', 'job' ])],
                                        remainder='passthrough')
        transformer.fit(self.features)
        self.features= transformer.transform(self.features)
        self.target = self.target.map({
    'Chipinge': 0,
    'Kwekwe': 1,
    'Karoi': 2,
    'Plumtree': 3,
    'Chiredzi': 4,
    'Rusape': 5,
    'Shurugwi': 6,
    'Masvingo': 7,
    'Gokwe': 8,
    'Mutare': 9,
    'Victoria Falls': 10,
    'Harare': 11,
    'Hwange': 12,
    'Bulawayo': 13,
    'Gweru': 14,
    'Marondera': 15,
    'Chivhu': 16,
    'Beitbridge': 17,
    'Chimanimani': 18,
    'Kadoma': 19,
    'Kariba': 20,
    'Nyanga': 21,
    'Zvishavane': 22
})
       
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

    
    
 


