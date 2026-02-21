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
    -   Add your OpenAI API Key: `OPENAI_API_KEY=sk-...` (Optional, can also be entered in the UI).

## Running the Application

To start the application, run:

```bash
streamlit run app.py
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
