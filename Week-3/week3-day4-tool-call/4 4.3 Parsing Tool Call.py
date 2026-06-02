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

prompt = st.chat_input("Try: Add a recipe for spaghetti, call the tool")

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

    st.write("")    
    st.write("Display the entire response object to see its structure:")    

    reply = response.json()
    st.write(reply)

    message = response.json()["message"]

    # Check if the model decided to call a tool
    if message.get("tool_calls"):
        args = message["tool_calls"][0]["function"]["arguments"]

        obj = {}
        for key, value in args.items():
            if isinstance(value, str):
                try:
                    obj[key] = json.loads(value)
                except json.JSONDecodeError:
                    obj[key] = value
            else:
                obj[key] = value
        
        st.write("")    
        st.write("Display the parsed object.")    
        st.json(obj)

    else:
        # Model answered directly without calling a tool
        with st.chat_message("assistant"):
            st.markdown(message["content"])

  
    