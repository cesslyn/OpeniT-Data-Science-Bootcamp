import streamlit as st
import requests
import json

st.set_page_config(page_title="Tool Calling", layout="wide")

st.title("Tool Calling")

# --- Define the tools ---

def string_processor(params):

    input_string = params["input"]
    # Example processing: reverse the string and convert to uppercase    
    return input_string[::-1].upper()

tools = [
    {
        "type": "function",
        "function": {
            "name": "string_processor",
            "description": "Processes a string and returns a modified version of it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The string you want to process"
                    }
                },
                "required": ["input"]
            }
        }
    }
]

prompt = st.chat_input("Try: Process 'phi4-mini' using the tool")

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

        result = string_processor(obj)
        st.write("")    
        st.write("Display the result of the tool function call.")    
        st.markdown(result)

    else:
        # Model answered directly without calling a tool
        with st.chat_message("assistant"):
            st.markdown(message["content"])

  
    