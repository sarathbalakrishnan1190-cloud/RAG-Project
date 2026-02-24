import time
import json
from typing import Generator, List, Dict

import requests
import streamlit as st

st.set_page_config(page_title="DocuMind Enterprise", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.title("📚 DocuMind Enterprise")
st.markdown("*Internal Corporate Knowledge Assistant*")


if "messages" not in st.session_state:
    
    st.session_state.messages = []  

if "last_task_id" not in st.session_state:
    
    st.session_state.last_task_id = None  


with st.sidebar:
    st.header("⚙️ Configuration")

    
    health_placeholder = st.empty()
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            health_data = health.json()
            with health_placeholder.container():
                st.success("✅ Backend Connected")
                st.caption(
                    f"Service: {health_data.get('service', 'DocuMind')} · "
                    f"Qdrant: {health_data.get('components', {}).get('qdrant', 'OK')}"
                )
        else:
            health_placeholder.error("❌ Backend Error")
    except Exception:
        health_placeholder.error("❌ Backend Offline")

    st.divider()

    
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files to the knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDFs at once.",
    )

    if uploaded_files:
        with st.expander("Selected files", expanded=False):
            for f in uploaded_files:
                st.write(f"• {f.name} ({round(len(f.getbuffer()) / 1024, 1)} KB)")

    
    start_upload = st.button("🚀 Upload & Process", use_container_width=True)
    clear_files = st.button("🧹 Clear selection", use_container_width=True)

    if clear_files:
        
        st.info("To clear the selected files, click the 'Rerun' button in the top-right menu.")

    if uploaded_files and start_upload:
        progress = st.progress(0, text="Preparing files...")
        status_area = st.empty()

        files = []
        total = len(uploaded_files)
        for idx, f in enumerate(uploaded_files, start=1):
            progress.progress(idx / total, text=f"Buffering {f.name}...")
            files.append(("files", (f.name, f.getbuffer(), "application/pdf")))

        try:
            progress.progress(1.0, text="Uploading to server...")
            response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
            if response.status_code == 200:
                result = response.json()
                task_id = result.get("task_id")
                st.session_state.last_task_id = task_id
                status_area.success(f"✅ Upload started! Task ID: `{task_id}`")
                status_area.info("Documents are being processed in the background.")

                
                with st.expander("Background task status", expanded=False):
                    st.caption("This checks the Celery task status once per refresh.")
                    if task_id:
                        try:
                            task_status = requests.get(f"{API_URL}/status/{task_id}", timeout=5)
                            if task_status.ok:
                                data = task_status.json()
                                st.write(f"Status: **{data.get('status', 'unknown')}**")
                            else:
                                st.write("Unable to fetch task status right now.")
                        except Exception:
                            st.write("Task status endpoint not reachable.")
            else:
                status_area.error(f"Upload failed: {response.text}")
        except Exception as e:
            status_area.error(f"Error: {str(e)}")
        finally:
            progress.empty()

    st.divider()

    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    
    recent_questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if recent_questions:
        st.subheader("🕑 Recent questions")
        for q in recent_questions[-5:][::-1]:
            if st.button(q, key=f"recent_{q[:32]}", help="Ask this again"):
                st.session_state.prompt_prefill = q
                st.rerun()


st.subheader("💬 Chat")

example_cols = st.columns(3)
examples = [
    "Summarize the latest policy updates.",
    "What are the key risks mentioned?",
    "List action items.",
]
for col, example in zip(example_cols, examples):
    if col.button(example):
        st.session_state.prompt_prefill = example
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt_prefill = st.session_state.pop("prompt_prefill", None)
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
        "history": history,
    }

    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with requests.post(
                f"{API_URL}/chat/stream",
                json=payload,
                stream=True,
                headers={"Content-Type": "application/json"},
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
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                else:
                    st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        st.info("Make sure the backend server is running: `uvicorn server:app --reload`")
