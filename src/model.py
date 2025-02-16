import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from data_loader import DataLoader
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
        
        self.attackerModel = RandomForestRegressor()
        self.defenderModel = RandomForestRegressor()
        self.midfielderModel = RandomForestRegressor()
        self.goalkeeperModel = RandomForestRegressor()
        self.midAttackerModel = RandomForestRegressor()
        self.midDefenderModel = RandomForestRegressor()
        
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
                "n_estimators": [64,100,128,150,200,300,500],
                "max_features": [int(len(features.columns)/3),int(np.sqrt(len(features.columns))),int(np.log2(len(features.columns))+1)],
                "bootstrap":[True,False],
                "oob_score":[True],
                
            }
            model = GridSearchCV(model,param_grid,verbose=2,n_jobs=-1)
            
            model.fit(X_train,y_train)
            return model.best_estimator_,X_test,y_test
        
        else:
            raise ValueError("Incorrect position provided.")