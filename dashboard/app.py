import streamlit as st
import pandas as pd

st.title("Toronto Maple Leafs 82-Game Outcome Predictor")

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv('../data/clean/predictions.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['date', 'opponent', 'p_win'])

preds = load_predictions()
if preds.empty:
    st.write("No predictions available yet.")
else:
    st.dataframe(preds)
