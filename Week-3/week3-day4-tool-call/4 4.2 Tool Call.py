import streamlit as st
import requests
import json

st.set_page_config(page_title="Tool Calling", layout="wide")

st.title("Tool Calling")

# --- Define the tools ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_recipe",
            "description": "Save a recipe with its details",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "The name of the recipe"
                    },
                    "ingredients": {
                        "type": "array",                        
                        "items": {"type": "string"}
                    },
                    "instructions": {
                        "type": "array",                        
                        "items": {"type": "string"}                       
                    }
                },
                "required": ["recipe_name", "ingredients", "instructions"]
            }
        }
    }
]

prompt = st.chat_input("Try: Add a recipe for spaghetti")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/chat", json=payload)
    
    reply = response.json()
    st.write(reply)

