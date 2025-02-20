import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import sys
import os



# Setup paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT_DIR, 'src'))
from data_loader import DataLoader

# Load data
data = DataLoader()

# Load models
attacker_model = load(os.path.join(ROOT_DIR, 'models', 'attacker_model.pkl'))
defender_model = load(os.path.join(ROOT_DIR, 'models', 'defender_model.pkl'))
goalkeeper_model = load(os.path.join(ROOT_DIR, 'models', 'goalkeeper_model.pkl'))
mid_attacker_model = load(os.path.join(ROOT_DIR, 'models', 'mid_attacker_model.pkl'))
mid_defender_model = load(os.path.join(ROOT_DIR, 'models', 'mid_defender_model.pkl'))

# Function to get position-based attributes
position_attributes = {
    "attacker": data.attacker()[0],
    "defender": data.defender()[0],
    "goalkeeper": data.goalkeeper()[0],
    "mid attacker": data.mid_attacker()[0],
    "mid defender": data.mid_defender()[0]
}

def predict_potential(position, to_pred):
    model_dict = {
        "attacker": attacker_model,
        "defender": defender_model,
        "goalkeeper": goalkeeper_model,
        "mid attacker": mid_attacker_model,
        "mid defender": mid_defender_model
    }
    
    model = model_dict.get(position)
    
    if model is None:
        st.error("Model for this position is not available yet.")
        return None
    
    to_pred = np.array(to_pred).reshape(1, -1)

    predicted_potential = model.predict(to_pred)
    return predicted_potential[0]

# 🎯 **Streamlit UI**
st.title("⚽ Football Talent Scout")
st.markdown("Predict a player's potential based on attributes!")

st.sidebar.header("Player Attributes")

# Store selected position in session state
if "selected_position" not in st.session_state:
    st.session_state.selected_position = "attacker"

# Dropdown for position
position = st.sidebar.selectbox(
    "Select Position:",
    list(position_attributes.keys()),
    index=list(position_attributes.keys()).index(st.session_state.selected_position),
    key="position_select"
)

# If position changes, update session state
if position != st.session_state.selected_position:
    st.session_state.selected_position = position
    st.rerun()  # 🔄 Force UI update

# Fixed attributes
age = st.sidebar.number_input("Age", min_value=16, max_value=45, value=25, step=1)


# Get dynamic attributes for the selected position
selected_attributes = position_attributes[position]
print(selected_attributes)
selected_attributes = selected_attributes[1:]

# Create sliders dynamically
skills = [st.sidebar.slider(attr, 0, 100, 70, key=f"{position}_{attr}") for attr in selected_attributes]

# Prepare input features
to_pred = [age] + skills

# Prediction Button
if st.sidebar.button("Predict Potential"):
    potential = predict_potential(position, to_pred)
    if potential > 0:
        st.success(f"Predicted Potential: {potential:.2f}")
