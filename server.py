from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel
from typing import List, Optional
from celery.result import AsyncResult
from celery_worker import process_documents_task, celery_app
from rag_engine import RAGEngine 
import os
import shutil
import uuid
from dotenv import load_dotenv


load_dotenv()


app = FastAPI(title="DocuMind Enterprise API")

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
    # Create a temp directory for uploads
    upload_dir = os.path.join(os.getcwd(), "temp_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_file_paths = []
    try:
        for file in files:
            # Generate unique filename to avoid collisions
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            saved_file_paths.append(file_path)

        task = process_documents_task.delay(saved_file_paths)

        return {"task_id": task.id, "status": "processing"}
    except Exception as e:
        for path in saved_file_paths:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/status/{task_id}")
def get_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None 
    }

@app.post("/chat/", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        # Convert dict history to LangChain message format if needed
        # Assuming request.history is [{"role": "user", "content": "..."}, ...]
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
    
@app.get("/")
def read_root():
    return {"message": "DocuMind Enterprise API is working"}