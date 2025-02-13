import numpy as np
import pandas as pd
from joblib import load
from data_loader import DataLoader


attacker_scaler = load("models/attacker scaler.pkl")
attacker_model = load("models/attacker model.pkl")
attacker_poly_converter = load("models/attacker poly_converter.pkl")


data = DataLoader()
attackers_features, _ = data.attacker()

position = input("Enter the position of the player: ")

if position == "attacker":

    to_pred = []

    for feature in attackers_features:
        value = float(input(f"{feature}: "))
        to_pred.append(value)


    to_pred = attacker_poly_converter.transform([to_pred])
    to_pred = attacker_scaler.transform(to_pred) 
    
    to_pred = 40 + (60 / (1 + np.exp(-0.08 * (to_pred - 90)))) # scaling for younger player help from ChatGPT

    predicted_potential = attacker_model.predict(to_pred)[0]


    print(f"Predicted Potential: {predicted_potential:.2f}")
