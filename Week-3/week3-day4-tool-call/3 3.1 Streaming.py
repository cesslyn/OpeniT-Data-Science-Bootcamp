import streamlit as st
import requests
import json

st.set_page_config(page_title="Ollama Chat", layout="wide")

st.title("Assistant Conversation")

# We will store the conversation history in Streamlit's session state, which allows us to keep track of messages across user interactions.

if "messages" not in st.session_state:
    st.session_state.messages = [
        # Also, initialize the system prompt
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Iterate through the conversation history
for msg in st.session_state.messages:
    # display each message in the appropriate format based on its role (system, user, assistant)
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# this is where the user will type their message. st.chat_input() is a built-in Streamlit component that creates a text input and automatically scrolls to the bottom of the page when new messages are added.
prompt = st.chat_input("Try this: Say \"meow\" verbatim.")



# run this logic every time the user submits a new message
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    payload = {
        "model": "phi4-mini",
        "messages": st.session_state.messages,
        #set the stream to true
        "stream": True
    }

    with st.chat_message("assistant"): placeholder = st.empty()
    
    complete_response = ""

    with requests.post("http://localhost:11434/api/chat", json=payload, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                text_chunk = chunk.get("message", {}).get("content", "")
                   # content = chunk["message"]["content"]
                #st.write(chunk)
                # Collect the reponse in this variable
                complete_response += text_chunk
                # Update the placeholder
                placeholder.markdown(complete_response + "▌")

    # Last update to the placeholder
    placeholder.markdown(complete_response)
    # Add the complete assistant message to the session state
    st.session_state.messages.append({"role": "assistant", "content": complete_response})


# Activity 3: What does the streaming response look like? Can you find the content of the assistant's reply in the response?