import streamlit as st
from lightkurve import search_lightcurve
import joblib
import pandas as pd
import numpy as np
import openai
import os

st.set_page_config(page_title="Exoplanet AI Explorer", layout="wide")
st.title("🔭 Exoplanet Discovery with AI")

# Load ML model
model = joblib.load("planet_classifier.pkl")

# OpenAI key input
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# TIC input
tic_id = st.text_input("Enter TIC ID (e.g. 307210830):")

def extract_features_from_lc(lc):
    folded = lc.fold(period=2.0)
    depth = np.nanmin(folded.flux)
    duration = 0.2
    snr = (1 - depth) / np.nanstd(folded.flux)
    return [depth, duration, snr]

if tic_id and st.button("Analyze"):
    st.write("🔍 Fetching data...")
    try:
        search_result = search_lightcurve(f"TIC {tic_id}", mission="TESS").download()
        search_result = search_result.remove_nans().normalize()
        folded = search_result.fold(period=2.0)
        folded.scatter(label="Folded Light Curve")

        st.write("📈 Light curve plotted.")

        # Feature extraction
        features = extract_features_from_lc(search_result)
        pred = model.predict([features])[0]
        label = "✅ Likely Exoplanet" if pred == 1 else "❌ Likely Not a Planet"
        st.subheader(f"🧠 AI Prediction: {label}")

        if openai_api_key:
            openai.api_key = openai_api_key
            prompt = f"Explain the following exoplanet signal to a student:\nDepth: {features[0]:.4f}\nDuration: {features[1]:.2f}\nSNR: {features[2]:.2f}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            explanation = response['choices'][0]['message']['content']
            st.write("🤖 AI Explanation:")
            st.success(explanation)
        else:
            st.info("Enter your OpenAI key to get explanation.")
    except Exception as e:
        st.error(f"Error: {e}")
