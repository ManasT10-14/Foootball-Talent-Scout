from model import Model
from sklearn.metrics import root_mean_squared_error,r2_score
from joblib import load,dump


model = Model()

positions = ["attacker","defender","midfielder","goalkeeper","mid_attacker","mid_defender"]
metric_data = ""
for position in positions:
    best_model,X_test,y_test = model.model(position)
    potential_predicted = best_model.predict(X_test)
    
    RMSE = root_mean_squared_error(y_test,potential_predicted)
    R2_score = r2_score(y_test,potential_predicted)
    metric_data+= f"{position} Model:\nRMSE: {RMSE}\nR2_score: {R2_score}\n\n"

    dump(best_model,f"../models/{position}_model.pkl")
with open("../models/metric.txt","w") as file:
    file.write(metric_data)