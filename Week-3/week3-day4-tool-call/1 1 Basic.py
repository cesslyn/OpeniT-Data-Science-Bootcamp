import streamlit as st
import requests
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="Ollama Chat", layout="wide")

st.title("Basic")

prompt = st.text_input("Ask phi4 something...")

messages = [
    {"role": "system", "content": "The student does not know how to create a system prompt yet."},
    {"role": "user", "content": prompt}
]

payload = {
    "model": "phi4-mini",
    "messages": messages,
    "stream": False
}

if st.button("Send") and prompt:
    response = requests.post( "http://localhost:11434/api/chat", json=payload)

    components.html(f"<script>console.log({json.dumps(response.json())})</script>", height=0)

    reply = response.json()
    st.write(reply["message"]["content"])    
    #st.write(reply)

# Activity 1: What does the response look like? Can you find the content of the assistant's reply in the response? make it so that the st.write() only shows the assistant's reply, not the entire response object.

# Activity 2: Add a system prompt that tells the assistant to be a cat that says meow, or a pirate that say Arrg. How does that change the response?