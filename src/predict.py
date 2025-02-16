import numpy as np
import pandas as pd
from joblib import load
from data_loader import DataLoader


attacker_model = load("models/attacker model.pkl")


data = DataLoader()
attackers_features, _ = data.attacker()

position = input("Enter the position of the player: ")

if position == "attacker":

    to_pred = []

    for feature in attackers_features:
        value = float(input(f"{feature}: "))
        to_pred.append(value)

    
    predicted_potential = attacker_model.predict([to_pred])[0]


    print(f"Predicted Potential: {predicted_potential:.2f}")
