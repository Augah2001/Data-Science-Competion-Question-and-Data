from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np



class Predict_ML:

    def __init(self,data):
        self.data = data
        self.features = ['gender','country',
       'is_employed', 'location', 'loan_amount', 'number_of_defaults',
       'outstanding_balance', 'interest_rate', 'age', 'number_of_defaults.1',
       'remaining term', 'salary']