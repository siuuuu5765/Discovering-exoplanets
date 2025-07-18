import streamlit as st
from lightkurve import search_lightcurve
import joblib
import pandas as pd
import numpy as np
import openai

# --- SETUP ---
st.set_page_config(page_title="Exoplanet AI Explorer", layout="wide")
st.title("🪐 Exoplanet Discovery with AI")

# --- Load ML Model ---
model = joblib.load("planet_classifier.pkl")

# --- OpenAI Key Input ---
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# --- TIC Input ---
tic_id = st.text_input("Enter TIC ID (e.g. 307210830):")

# --- Extract Features ---
def extract_features_from_lc(lc):
    folded = lc.fold(period=2.0)
    depth = np.nanmin(folded.flux)
    duration = 0.2
    snr = (1 - depth) / np.nanstd(folded.flux)
    return [depth, duration, snr]

# --- Analyze Button ---
if tic_id and st.button("Analyze"):
    st.write("🔍 Fetching data...")
    try:
        search_result = search_lightcurve(f"TIC {tic_id}", mission="TESS").download()
        search_result = search_result.remove_nans().normalize()
        folded = search_result.fold(period=2.0)
        folded.scatter(label="Folded Light Curve")
        st.write("✅ Light curve plotted.")

        features = extract_features_from_lc(search_result)
        pred = model.predict([features])[0]
        label = "🌍 Likely Exoplanet" if pred == 1 else "❌ Likely Not a Planet"
        st.subheader(f"🧠 AI Prediction: {label}")

        if openai_api_key:
            openai.api_key = openai_api_key
            prompt = f"Explain the following exoplanet signal to a student:\nDepth: {features[0]:.4f}\nDuration: {features[1]:.2f}\nSNR: {features[2]:.2f}"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                explanation = response['choices'][0]['message']['content']
                st.write("📘 Explanation:")
                st.success(explanation)
            except Exception as e:
                st.error(f"⚠️ OpenAI error: {e}")
        else:
            st.info("Enter your OpenAI API key to receive an explanation.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Chatbot Section ---
st.markdown("---")
st.subheader("💬 Ask the AI about exoplanets")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are an expert in exoplanets. Explain in simple terms."}]

user_input = st.chat_input("Ask me anything about exoplanets...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    if openai_api_key:
        openai.api_key = openai_api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.chat_history
            )
            reply = response["choices"][0]["message"]["content"]
        except Exception as e:
            reply = f"⚠️ Error: {e}"
    else:
        reply = "⚠️ Please provide your OpenAI API key above."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# Show full conversation
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])