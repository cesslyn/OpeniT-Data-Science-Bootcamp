import streamlit as st
import requests
import json

st.set_page_config(page_title="Tool Calling", layout="wide")

st.title("Tool Calling")

#https://docs.ollama.com/capabilities/tool-calling?utm_source=chatgpt.com

# --- Define the tools ---
# This tells the model what functions it is allowed to call
# {
#   "city": "{city name}" 
# }
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

prompt = st.chat_input("Try: What is the weather in Dallas?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "model": "phi4-mini",
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/chat", json=payload)
    
    reply = response.json()
    st.write(reply)