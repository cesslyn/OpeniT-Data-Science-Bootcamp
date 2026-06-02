import streamlit as st
import fastapi_poe as fp
import asyncio

#https://creator.poe.com/docs/external-applications/external-application-guide#querying-poe-bots-via-an-api-key

st.set_page_config(page_title="Poe Chat", layout="wide")

st.title("Poe")

# --- Config ---
API_KEY = st.sidebar.text_input("Poe API Key", type="password")
BOT_NAME = st.sidebar.selectbox("Bot", ["Claude-3.7-Sonnet", "GPT-4o", "Gemini-Pro", "DeepSeek-V3"])

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Async streaming call ---
async def stream(api_key, bot_name, messages, placeholder):
    poe_messages = [
        fp.ProtocolMessage(role=m["role"], content=m["content"])
        for m in messages
    ]
    full_text = ""
    async for chunk in fp.get_bot_response(messages=poe_messages, bot_name=bot_name, api_key=api_key):
        full_text += chunk.text
        placeholder.markdown(full_text + "▌")
    return full_text

# --- Chat ---
prompt = st.chat_input("Ask something...")

if prompt:
    if not API_KEY:
        st.warning("Enter your Poe API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            reply = asyncio.run(stream(API_KEY, BOT_NAME, st.session_state.messages, placeholder))
            placeholder.markdown(reply)
        except Exception as e:
            st.error(f"Bot error: {e}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": reply})
