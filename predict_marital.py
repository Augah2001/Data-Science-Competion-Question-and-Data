from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

model =     RandomForestClassifier()

class Predict_ML:
   

    def __init__(self, data, model=model):
        self.predict_values = data.loc[data.marital_status.isna(),['gender','country',
       'is_employed', 'location', 'loan_amount', 'number_of_defaults',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.features = data.loc[~data.marital_status.isna(),['gender','country',
       'is_employed', 'location', 'loan_amount', 'number_of_defaults',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.target = data.loc[~data.marital_status.isna(),'marital_status']
        self.model = model

    def preprocess(self):
        self.features.location.fillna('Unknown', inplace=True)
        one_hot = OneHotEncoder()
        le = LabelEncoder()

        transformer = ColumnTransformer([('one_hot', one_hot, ['gender', 'country', 'is_employed', 'location'])],
                                        remainder='passthrough')
        self.features= transformer.fit_transform(self.features)
        le.fit(['single', 'married', 'divorced'])
        self.target = le.transform(self.target)
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
   

    def __init__(self, data, model=model):
        self.predict_values = data.loc[data.job.isna(),['gender','country',
       'is_employed', 'location', 'loan_amount', 'number_of_defaults','marital_status',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.features = data.loc[~data.job.isna(),['gender','marital_status','country',
       'is_employed', 'location', 'loan_amount', 'number_of_defaults',
       'outstanding_balance', 'interest_rate', 'age',
       'remaining term', 'salary']]
        self.target = data.loc[~data.job.isna(),'job']
        self.model = model

    def preprocess(self):
        one_hot = OneHotEncoder()
        le = LabelEncoder()

        transformer = ColumnTransformer([('one_hot', one_hot, ['gender', 'marital_status', 'country', 'is_employed', 'location'])],
                                        remainder='passthrough')
        self.features= transformer.fit_transform(self.features)
        le.fit(['Teacher', 'Nurse', 'Doctor', 'Data Analyst', 'Software Developer',
       'Accountant', 'Lawyer', 'Engineer','Data Scientist'])
        self.target = le.transform(self.target)
        self.predict_values = transformer.fit_transform(self.predict_values)
        
        return self.features, self.target
    

    def train(self):
        features, target = self.preprocess()
        
        self.model.fit(features, target)

        
        return self.model
    
    def predict(self):
        
        preds = self.model.predict(self.predict_values)
        return preds

    
    
 


