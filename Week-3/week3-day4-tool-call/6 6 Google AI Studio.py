import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Google AI Studio", layout="wide")

st.title("Google AI Studio — Gemini")

# --- Config ---
API_KEY = st.sidebar.text_input("Google AI Studio API Key", type="password")
BOT_NAME = st.sidebar.selectbox("Model", ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"])

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat ---
prompt = st.chat_input("Ask something...")

if prompt:
    if not API_KEY:
        st.warning("Enter your Google AI Studio API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(BOT_NAME)

    history = [
        {"role": m["role"], "parts": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    chat = model.start_chat(history=history)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        for chunk in chat.send_message(prompt, stream=True):
            reply += chunk.text
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
