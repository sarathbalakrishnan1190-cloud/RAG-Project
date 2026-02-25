import time
import json
from typing import Generator, List, Dict

import requests
import streamlit as st

st.set_page_config(
    page_title="DocuMind Enterprise",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 1. Hide unwanted Streamlit Toolbar items (Deploy, Menu) but keep Toggle */
    [data-testid="stToolbar"] {
        visibility: visible !important;
        display: block !important;
        background: transparent !important;
    }
    
    /* Hide everything in toolbar except the expand button */
    [data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]) {
        display: none !important;
    }

    /* 2. Hide decoration line */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* 3. Permanent Sidebar Toggle Fix - SELECTIVE VISIBILITY */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Expand Button (when sidebar is closed) */
    [data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        display: flex !important;
        background-color: #00a67e !important;
        color: white !important;
        border-radius: 8px !important;
        top: 15px !important;
        left: 15px !important;
        position: fixed !important;
        z-index: 1000001 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
        width: 45px !important;
        height: 45px !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stExpandSidebarButton"] svg {
        fill: white !important;
    }

    /* Collapse Button (when sidebar is open) */
    [data-testid="stSidebarCollapseButton"] {
        background-color: #00a67e !important;
        color: white !important;
        border-radius: 6px !important;
        margin: 10px !important;
    }

    [data-testid="stSidebarCollapseButton"] svg {
        fill: white !important;
    }

    footer {
        display: none !important;
    }
    
    .stApp {
        background-color: #212121;
        color: #ececec;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, inherit;
    }
    
    h1 {
        text-align: center !important;
        width: 100% !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        width: 300px !important;
    }
    
    .stButton>button {
        width: 100%;
        background-color: transparent;
        color: white;
        border: 1px solid #4d4d4d;
        border-radius: 8px;
        text-align: left !important;
        padding: 12px 15px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 14px;
        margin-bottom: 4px;
    }
    
    .stButton>button:hover {
        background-color: #2a2b32 !important;
        border-color: #666;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }

    .stChatInputContainer {
        padding-bottom: 2.5rem;
        background-color: transparent !important;
    }
    
    div[data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid #4d4d4d !important;
        background-color: #2f2f2f !important;
        box-shadow: 0 0 20px rgba(0,0,0,0.2);
        transition: border-color 0.3s ease;
    }
    
    div[data-testid="stChatInput"]:focus-within {
        border-color: #00a67e !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stChatMessage {
        background-color: transparent !important;
        padding: 2rem 0 !important;
        border-bottom: 0.5px solid #3d3d3d;
        animation: fadeIn 0.4s ease-out;
    }
    
    div[data-testid="stChatMessageAssistant"] {
        background-color: #2f2f2f !important;
    }

    .tile-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 2rem;
    }
    
    .action-tile {
        background: #2f2f2f;
        border: 1px solid #4d4d4d;
        border-radius: 12px;
        padding: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .action-tile:hover {
        background: #3e3e3e;
        border-color: #666;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

if "chats" not in st.session_state:
    st.session_state.chats = {"Enterprise Knowledge": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Enterprise Knowledge"

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.rerun()
    
    st.divider()
    
    st.caption("HISTORY")
    for chat_name in reversed(list(st.session_state.chats.keys())):
        is_active = chat_name == st.session_state.current_chat
        btn_label = f"💬 {chat_name}"
        if is_active:
            btn_label = f"✨ {chat_name}"
        
        if st.button(btn_label, key=f"chat_{chat_name}", use_container_width=True):
            st.session_state.current_chat = chat_name
            st.rerun()

    st.divider()
    
    with st.expander("📂 Knowledge Base", expanded=False):
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Add PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("🔥 Process Knowledge", use_container_width=True):
                progress = st.progress(0)
                files = []
                for f in uploaded_files:
                    files.append(("files", (f.name, f.getbuffer(), "application/pdf")))
                
                try:
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                    if response.status_code == 200:
                        st.success("✨ Knowledge Base Updated!")
                    else:
                        st.error("❌ Link failure.")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    progress.empty()

    st.divider()
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=1)
        if health.status_code == 200:
            st.caption("🟢 System Ready")
        else:
            st.caption("🔴 Backend Issue")
    except:
        st.caption("⚪ Offline")

st.title(f"DocuMind: {st.session_state.current_chat}")

current_messages = st.session_state.chats[st.session_state.current_chat]

if not current_messages:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white;'>How can I help you today?</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Summarize recent policy updates", key="tile1", use_container_width=True):
            st.session_state.temp_prompt = "Summarize the latest policy updates."
            st.rerun()
        if st.button("🔍 Analyze document risks", key="tile2", use_container_width=True):
            st.session_state.temp_prompt = "What are the key risks mentioned in these documents?"
            st.rerun()
    with col2:
        if st.button("📋 Generate action item list", key="tile3", use_container_width=True):
            st.session_state.temp_prompt = "Extract action items from the current context."
            st.rerun()
        if st.button("💡 Creative ideas for strategy", key="tile4", use_container_width=True):
            st.session_state.temp_prompt = "Suggest 5 creative improvements for our current strategy."
            st.rerun()
else:
    for message in current_messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

prompt = st.chat_input("Message DocuMind...")
if "temp_prompt" in st.session_state:
    prompt = st.session_state.pop("temp_prompt")

if prompt:
    current_messages.append({"role": "user", "content": prompt})
    st.rerun()

if current_messages and current_messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in current_messages[:-1]
        ]
        
        try:
            user_input = current_messages[-1]["content"]
            with requests.post(
                f"{API_URL}/chat/stream",
                json={"question": user_input, "history": history},
                stream=True,
                headers={"Content-Type": "application/json"},
                timeout=60
            ) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data: "):
                                try:
                                    data = json.loads(decoded_line[6:])
                                    if "token" in data:
                                        full_response += data["token"]
                                        message_placeholder.markdown(full_response + "▌")
                                except:
                                    pass
                    message_placeholder.markdown(full_response)
                    current_messages.append({"role": "assistant", "content": full_response})
                    st.rerun()
                else:
                    st.error(f"Connection failed ({response.status_code})")
        except Exception as e:
            st.error(f"Error: {e}")
