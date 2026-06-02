import streamlit as st
import requests
import json

st.set_page_config(page_title="Ollama Chat", layout="wide")

st.title("Built in Chat UI")

st.text_input("This is a text input")

with st.chat_message("system"): st.markdown("You are a helpful assistant.")

with st.chat_message("user"): st.markdown("This is a user message.")

with st.chat_message("assistant"): st.markdown("This is an assistant message.")

st.chat_input("This is pulled down to the bottom of the page automatically")