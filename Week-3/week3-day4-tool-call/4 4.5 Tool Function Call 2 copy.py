import streamlit as st
import requests
import json

st.set_page_config(page_title="Tool Calling", layout="wide")

st.title("Tool Calling")

# --- Define the tools ---

def country_flag(params):
    country = params["country"]
    response = requests.get(f"https://restcountries.com/v3.1/name/{country}?fields=flags")
    data = response.json()
    return data[0]["flags"]

tools = [
    {
        "type": "function",
        "function": {
            "name": "country_flag",
            "description": "shows the flag of a country",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "The country you want to get the flag for"
                    }
                },
                "required": ["country"]
            }
        }
    }
]

prompt = st.chat_input("Try: Get the flag of the Philippines using the tool")

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

        result = country_flag(obj)
        st.write("")
        st.write("Display the result of the tool function call.")
        st.json(result)
        st.image(result["png"], caption=result.get("alt", ""))

    else:
        # Model answered directly without calling a tool
        with st.chat_message("assistant"):
            st.markdown(message["content"])

  
    