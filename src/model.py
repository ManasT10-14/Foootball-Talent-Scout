import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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
            
            polyfeatures = PolynomialFeatures(degree=2,include_bias=False)
            X_train = polyfeatures.fit_transform(X_train)    
            X_test = polyfeatures.transform(X_test)  
            
            scaler = MinMaxScaler(feature_range=(0,100))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            param_grid = {
                "alpha":[0.0001,0.001,0.01,0.1,1,10,100],
                "l1_ratio":[0.1,0.3,0.5,0.7,0.9,0.92,0.95,0.97,1]
            }
            model = GridSearchCV(model,param_grid,scoring={"RMSE": "neg_root_mean_squared_error", "R2": "r2"},cv=5,refit="RMSE")
            
            model.fit(X_train,y_train)
            return model.best_estimator_,scaler,X_test,y_test,polyfeatures
        
        else:
            raise ValueError("Incorrect position provided.")