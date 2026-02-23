# DocuMind Enterprise

DocuMind Enterprise is a strict corporate knowledge assistant built with Streamlit and LangChain.

## Setup

1.  **Prerequisites**: Python 3.8+ installed.
2.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    -   Create a `.env` file in the root directory.
    -   The engine prefers a HuggingFace token first (`HUGGINGFACEHUB_API_TOKEN`). If that is
        absent but you have an OpenAI key, the code will automatically use OpenAI's
        Chat model.
    -   You may also specify which HuggingFace model/repo to call by setting
        `HF_MODEL` (e.g. `google/flan-t5-small`). If omitted, a small FLAN model is
        used by default. The previous default (`google/flan-t5-large`) may no longer
        be available and resulted in 410 errors.
    -   **Important:** the token must be created with **inference scope** (or you
        will receive HTTP 410 errors). Create a new token on Hugging Face and enable
        the inference permission, or use an OpenAI key instead.
    -   Example `.env`:
        ```text
        OPENAI_API_KEY=sk-...
        # or HUGGINGFACEHUB_API_TOKEN=hf_...
        HF_MODEL=google/flan-t5-small
        ```
    -   You can also enter the OpenAI key (or override HF_MODEL) via the Streamlit
        sidebar at runtime.

## Running the Application

To start the application, run the backend and optionally the Streamlit UI:

```bash
# backend API
uvicorn server:app --reload

# in another shell (optional frontend)
streamlit run app_streamlit.py
```

## Usage

1.  Open the URL provided by Streamlit (usually `http://localhost:8501`).
2.  Enter your OpenAI API Key in the sidebar (if not set in `.env`).
3.  Upload PDF documents.
4.  Click "Process Documents".
5.  Start asking questions in the chat interface.

## Architecture

-   **`app.py`**: Streamlit frontend.
-   **`rag_engine.py`**: Core RAG logic (Document loading, splitting, vector store).
-   **`prompts.py`**: Strict system prompt configuration.
