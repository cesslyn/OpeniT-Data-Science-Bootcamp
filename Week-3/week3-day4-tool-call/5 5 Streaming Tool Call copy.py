import streamlit as st
import requests
import json

st.set_page_config(page_title="Book Finder", layout="wide")
st.title("Book Finder")


# TOOL FUNCTION
def search_books(params):
    query = params["query"]

    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"
    response = requests.get(url)
    data = response.json()

    books = data.get("items", [])

    results_text = ""

    for book in books:
        info = book.get("volumeInfo", {})

        title = info.get("title", "No title")
        authors = ", ".join(info.get("authors", ["Unknown"]))
        description = info.get("description", "No description")
        published = info.get("publishedDate", "Unknown")

        thumbnail = info.get("imageLinks", {}).get("thumbnail")

        results_text += f"""
**{title}**
- Authors: {authors}
- Published: {published}
- Description: {description[:200]}...

---
"""

        if thumbnail:
            st.image(thumbnail, caption=title)

    return results_text if results_text else "No books found."


def process_tool(params, tool_name):
    if tool_name == "search_books":
        return search_books(params)
    else:
        return f"Tool {tool_name} not found."


# TOOL SCHEMA
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": "shows details about the book",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]


# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant that finds and explains books."}
    ]


# CHAT HISTORY DISPLAY
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Try this: psychology books, fiction, sci-fi")


# GENERATION FUNCTION
def generate():
    payload = {
        "model": "llama3.2",
        "messages": st.session_state.messages,
        "tools": tools,
        "stream": True
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()

    complete_response = ""
    has_tool_calls = False

    with requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        stream=True
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)

                tool_calls = chunk.get("message", {}).get("tool_calls")


                if tool_calls:
                    has_tool_calls = True

                    for tool_call in tool_calls:
                        tool_call_id = tool_call["id"]
                        tool_call_name = tool_call["function"]["name"]
                        tool_call_args = tool_call["function"]["arguments"]

                        # parse arguments safely
                        obj = {}

                        for key, value in tool_call_args.items():
                            if isinstance(value, str):
                                try:
                                    obj[key] = json.loads(value)
                                except json.JSONDecodeError:
                                    obj[key] = value
                            else:
                                obj[key] = value

                        tool_response = process_tool(obj, tool_call_name)

                        st.session_state.messages.append({
                            "role": "tool",
                            "content": tool_response,
                            "tool_name": tool_call_name,
                            "tool_call_id": tool_call_id
                        })

                else:
                    complete_response += chunk["message"]["content"]
                    placeholder.markdown(complete_response + "▌")

    if has_tool_calls:
        generate()

    else:
        placeholder.markdown(complete_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": complete_response
        })


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    generate()