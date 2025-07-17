import streamlit as st
from lightkurve import search_lightcurve
import joblib
import pandas as pd
import numpy as np
import openai
import os
import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI()

def get_chat_response():
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.chat_history
        )
        reply = response.choices[0].message.content
        return reply
    except Exception as e:
        return f"Error: {e}"



st.set_page_config(page_title="Exoplanet AI Explorer", layout="wide")
st.title("Exoplanet Discovery with AI")

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

        st.write("Light curve plotted.")

        # Feature extraction
        features = extract_features_from_lc(search_result)
        pred = model.predict([features])[0]
        label = " Likely Exoplanet" if pred == 1 else "❌ Likely Not a Planet"
        st.subheader(f"🧠 AI Prediction: {label}")

        if openai_api_key:
            openai.api_key = openai_api_key
            prompt = f"Explain the following exoplanet signal to a student:\nDepth: {features[0]:.4f}\nDuration: {features[1]:.2f}\nSNR: {features[2]:.2f}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            explanation = response['choices'][0]['message']['content']
            st.write(" AI Explanation:")
            st.success(explanation)
        else:
            st.info("Enter your OpenAI key to get explanation.")
    except Exception as e:
        st.error(f"Error: {e}")
import openai

# --- OPENAI CHATBOT ---

st.markdown("---")
st.subheader(" Ask the AI about exoplanets")

# Load API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask me about exoplanets...")

if user_input:
    with st.spinner("Thinking..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

import openai

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=st.session_state.chat_history
)

assistant_message = response.choices[0].message.content


assistant_message = response.choices[0].message.content

def get_chat_response():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.chat_history
    )
    reply = response.choices[0].message.content
    return reply

st.session_state.chat_history.append({"role": "assistant", "content": reply})

# Display chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🔭 Exoplanet Discovery Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are an expert in exoplanets. Explain in simple terms."}]

def get_chat_response():
    user_input = st.text_input("Ask something about exoplanets:")
    if st.button("Ask") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.chat_history
            )
            reply = response["choices"][0]["message"]["content"]
        except Exception as e:
            reply = f"⚠️ Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.write("🧠 AI:", reply)

get_chat_response()
