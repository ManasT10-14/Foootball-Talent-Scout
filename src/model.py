import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from data_loader import DataLoader
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
class Model:
    def __init__(self):
        self.dataLoader = DataLoader()
        
        self.attackerFeatures, self.attackerData = self.dataLoader.attacker()
        self.defenderFeatures, self.defenderData = self.dataLoader.defender()
        self.midfielderFeatures, self.midfielderData = self.dataLoader.midfielder()
        self.goalkeeperFeatures, self.goalkeeperData = self.dataLoader.goalkeeper()
        self.midAttackerFeatures, self.midAttackerData = self.dataLoader.mid_attacker()
        self.midDefenderFeatures, self.midDefenderData = self.dataLoader.mid_defender()
        
        self.attackerModel = XGBRegressor(tree_method="hist",  # GPU-compatible method
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1 )
        self.defenderModel = XGBRegressor(tree_method="hist",  # GPU-compatible method
  # Speed boost using lower precision
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1 )
        self.midfielderModel = XGBRegressor(tree_method="hist",  # GPU-compatible method
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1)
        self.goalkeeperModel = XGBRegressor(tree_method="hist",
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1)
        self.midAttackerModel = XGBRegressor(tree_method="hist", 

    objective="reg:squarederror",
    random_state=42,
    n_jobs=1 )
        self.midDefenderModel = XGBRegressor(tree_method="hist",
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1)
        
        self.__models = {
            "attacker":[self.attackerFeatures,self.attackerData,self.attackerModel],
            "defender":[self.defenderFeatures,self.defenderData,self.defenderModel],
            "midfielder":[self.midfielderFeatures,self.midfielderData,self.midfielderModel],
            "goalkeeper":[self.goalkeeperFeatures,self.goalkeeperData,self.goalkeeperModel],
            "mid_attacker":[self.midAttackerFeatures,self.midAttackerData,self.midAttackerModel],
            "mid_defender":[self.midDefenderFeatures,self.midDefenderData,self.midDefenderModel]
        }
        
        
    def model(self,position):
        if position in self.__models.keys():
            data = self.__models[position][1]
            features = data[self.__models[position][0]]
            model = self.__models[position][2]

            
            print(f"Checking data alignment for position: {position}")
            print(f"Features shape: {features.shape}")
            print(f"Target shape: {data['potential'].shape}")
            
            X_train,X_test,y_train,y_test = train_test_split(features,data["potential"],test_size=0.1,random_state=42)  

            param_grid = {
                "n_estimators": [100,200,300,500],
                "learning_rate": [0.01,0.1],
                "max_depth": [3, 5,6,7,8], 
                "subsample": [0.8],  
                "colsample_bytree": [0.8], 
                "gamma": [0.1, 0.2],
                "reg_lambda": [1,6,7,10],  
                "reg_alpha": [1,5, 9,10],  
                "min_child_weight": [3, 5,6],  
                
            }
            model = GridSearchCV(model,param_grid,cv=3,verbose=2,n_jobs=-1,scoring="r2",refit=True)
            model.fit(X_train, y_train)
            print(model.best_estimator_)
            return model.best_estimator_,X_test,y_test
        
        else:
            raise ValueError("Incorrect position provided.")