import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from data_loader import DataLoader

class Model:
    def __init__(self):
        self.dataLoader = DataLoader()
        
        self.attackerFeatures, self.attackerData = self.dataLoader.attacker()
        self.defenderFeatures, self.defenderData = self.dataLoader.defender()
        self.midfielderFeatures, self.midfielderData = self.dataLoader.midfielder()
        self.goalkeeperFeatures, self.goalkeeperData = self.dataLoader.goalkeeper()
        self.midAttackerFeatures, self.midAttackerData = self.dataLoader.mid_attacker()
        self.midDefenderFeatures, self.midDefenderData = self.dataLoader.mid_defender()
        
        self.attackerModel = ElasticNet()
        self.defenderModel = ElasticNet()
        self.midfielderModel = ElasticNet()
        self.goalkeeperModel = ElasticNet()
        self.midAttackerModel = ElasticNet()
        self.midDefenderModel = ElasticNet()
        
    def attacker_model(self):
        X_train,X_test,y_train,y_test = train_test_split(self.attackerFeatures,self.attackerData["potential"],test_size=0.1,random_state=42)
        
        polyfeatures = PolynomialFeatures(degree=2,include_bias=False)
        X_train = polyfeatures.fit_transform(X_train)    
        X_test = polyfeatures.transform(X_test)  
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        param_grid = {
            "alpha":[0.0001,0.001,0.01,0.1,1,10,100],
            "l1_ratio":[0.1,0.3,0.5,0.7,0.9,0.92,0.95,0.97,1]
        }
        model = GridSearchCV(self.attackerModel,param_grid,scoring=["neg_root_mean_squared_error","r2"],cv=5)
        
        return model,polyfeatures,scaler,X_test,y_test
    
    def defender_model(self):
        X_train,X_test,y_train,y_test = train_test_split(self.defenderFeatures,self.defenderData["potential"],test_size=0.1,random_state=42)
        
        polyfeatures = PolynomialFeatures(degree=2,include_bias=False)
        X_train = polyfeatures.fit_transform(X_train)    
        X_test = polyfeatures.transform(X_test)  
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        param_grid = {
            "alpha":[0.0001,0.001,0.01,0.1,1,10,100],
            "l1_ratio":[0.1,0.3,0.5,0.7,0.9,0.92,0.95,0.97,1]
        }
        model = GridSearchCV(self.defenderModel,param_grid,scoring=["neg_root_mean_squared_error","r2"],cv=5)
        
        return model,polyfeatures,scaler,X_test,y_test
    
    def midfielder_model(self):
        X_train,X_test,y_train,y_test = train_test_split(self.midfielderFeatures,self.midfielderData["potential"],test_size=0.1,random_state=42)
        
        polyfeatures = PolynomialFeatures(degree=2,include_bias=False)
        X_train = polyfeatures.fit_transform(X_train)    
        X_test = polyfeatures.transform(X_test)  
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        param_grid = {
            "alpha":[0.0001,0.001,0.01,0.1,1,10,100],
            "l1_ratio":[0.1,0.3,0.5,0.7,0.9,0.92,0.95,0.97,1]
        }
        model = GridSearchCV(self.midfielderModel,param_grid,scoring=["neg_root_mean_squared_error","r2"],cv=5)
        
        return model,polyfeatures,scaler,X_test,y_test
    
    def goalkeeper_model(self):
        X_train,X_test,y_train,y_test = train_test_split(self.goalkeeperFeatures,self.goalkeeperData["potential"],test_size=0.1,random_state=42)
        
        polyfeatures = PolynomialFeatures(degree=2,include_bias=False)
        X_train = polyfeatures.fit_transform(X_train)    
        X_test = polyfeatures.transform(X_test)  
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        param_grid = {
            "alpha":[0.0001,0.001,0.01,0.1,1,10,100],
            "l1_ratio":[0.1,0.3,0.5,0.7,0.9,0.92,0.95,0.97,1]
        }
        model = GridSearchCV(self.goalkeeperModel,param_grid,scoring=["neg_root_mean_squared_error","r2"],cv=5)
        
        return model,polyfeatures,scaler,X_test,y_test
    
    def mid_attacker_model(self):
        X_train,X_test,y_train,y_test = train_test_split(self.midAttackerFeatures,self.midAttackerData["potential"],test_size=0.1,random_state=42)
        
        polyfeatures = PolynomialFeatures(degree=2,include_bias=False)
        X_train = polyfeatures.fit_transform(X_train)    
        X_test = polyfeatures.transform(X_test)  
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        param_grid = {
            "alpha":[0.0001,0.001,0.01,0.1,1,10,100],
            "l1_ratio":[0.1,0.3,0.5,0.7,0.9,0.92,0.95,0.97,1]
        }
        model = GridSearchCV(self.midAttackerModel,param_grid,scoring=["neg_root_mean_squared_error","r2"],cv=5)
        
        return model,polyfeatures,scaler,X_test,y_test
    
    def mid_defender_model(self):
        X_train,X_test,y_train,y_test = train_test_split(self.midDefenderFeatures,self.midDefenderData["potential"],test_size=0.1,random_state=42)
        
        polyfeatures = PolynomialFeatures(degree=2,include_bias=False)
        X_train = polyfeatures.fit_transform(X_train)    
        X_test = polyfeatures.transform(X_test)  
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        param_grid = {
            "alpha":[0.0001,0.001,0.01,0.1,1,10,100],
            "l1_ratio":[0.1,0.3,0.5,0.7,0.9,0.92,0.95,0.97,1]
        }
        model = GridSearchCV(self.midDefenderModel,param_grid,scoring=["neg_root_mean_squared_error","r2"],cv=5)
        
        return model,polyfeatures,scaler,X_test,y_test
    
     