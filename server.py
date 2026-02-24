from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import json
import logging
from dotenv import load_dotenv

load_dotenv()


USE_SYNC_UPLOAD = os.getenv("USE_SYNC_UPLOAD", "true").lower() in ("true", "1", "yes")

if not USE_SYNC_UPLOAD:
    from celery.result import AsyncResult
    from celery_worker import process_documents_task, celery_app

from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DocuMindAPI")

app = FastAPI(title="DocuMind Enterprise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()
class QueryRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []
class QueryResponse(BaseModel):
    answer: str
class TaskResponse(BaseModel):
    task_id: str
    status: str
@app.post("/upload", response_model=TaskResponse)
async def upload_documents(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    upload_dir = os.path.join(os.getcwd(), "temp_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    saved_file_paths = []
    try:
        for file in files:
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(upload_dir, unique_filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_file_paths.append(file_path)

        if USE_SYNC_UPLOAD:
            result = rag_engine.process_documents(saved_file_paths)
            for path in saved_file_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            return {"task_id": "sync", "status": "completed"}
        else:
            task = process_documents_task.delay(saved_file_paths)
            return {"task_id": task.id, "status": "processing"}
    except Exception as e:
        for path in saved_file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}")
def get_status(task_id: str):
    if USE_SYNC_UPLOAD and task_id == "sync":
        return {"task_id": task_id, "status": "SUCCESS", "result": "completed"}
    if not USE_SYNC_UPLOAD:
        task_result = AsyncResult(task_id, app=celery_app)
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result if task_result.ready() else None,
        }
    return {"task_id": task_id, "status": "unknown", "result": None}

@app.post("/chat/", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
       
        formatted_history = []
        history = request.history or []
        for msg in history:
            if msg["role"] == "user":
                formatted_history.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(("ai", msg["content"]))
        
        response = rag_engine.query(request.question, chat_history=formatted_history)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    Streaming chat endpoint for real-time token generation.
    """
    formatted_history = []
    history = request.history or []
    for msg in history:
        if msg.get("role") == "user":
            formatted_history.append(("human", msg.get("content")))
        elif msg.get("role") == "assistant":
            formatted_history.append(("ai", msg.get("content")))
    
    def event_generator():
        try:
            for token in rag_engine.stream_query(request.question, chat_history=formatted_history):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            print("REAL ERROR:", repr(e))   
            yield f"data: {json.dumps({'error': repr(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    """
    Detailed health check for production monitoring.
    """
    return {
        "status": "healthy",
        "service": "DocuMind Enterprise API",
        "components": {
            "rag_engine": "initialized",
            "qdrant": rag_engine.qdrant_url
        }
    }

@app.get("/")
def read_root():
    return {"message": "DocuMind Enterprise API is working. Use /health for details."}