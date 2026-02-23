import streamlit as st
import requests
import json
from typing import Generator

st.set_page_config(page_title="DocuMind Enterprise", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.title("📚 DocuMind Enterprise")
st.markdown("*Internal Corporate Knowledge Assistant*")


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("⚙️ Configuration")
    
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Error")
    except:
        st.error("❌ Backend Offline")
    
    
    
    st.divider()
    
    
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files to the knowledge base",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Upload & Process"):
        with st.spinner("Processing documents..."):
            files = [("files", (f.name, f.getbuffer(), "application/pdf")) for f in uploaded_files]
            try:
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    result = response.json()
                    task_id = result.get("task_id")
                    st.success(f"✅ Upload started! Task ID: {task_id}")
                    st.info("Documents are being processed in the background.")
                else:
                    st.error(f"Upload failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


st.subheader("💬 Chat")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about your documents..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]  
    ]
    
    payload = {
        "question": prompt,
        "history": history
    }
    
    
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with requests.post(
                f"{API_URL}/chat/stream",
                json=payload,
                stream=True,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode("utf-8")
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if "token" in data:
                                        token = data["token"]
                                        full_response += token
                                        message_placeholder.markdown(full_response + "▌")
                                    elif "error" in data:
                                        st.error(f"Error: {data['error']}")
                                        break
                                except json.JSONDecodeError:
                                    pass
                    
                    
                    message_placeholder.markdown(full_response)
                    
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        st.info("Make sure the backend server is running: `uvicorn server:app --reload`")
