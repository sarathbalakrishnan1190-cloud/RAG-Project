import os
from celery import Celery
from rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "documind_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
)


rag_engine=RAGEngine()

@celery_app.task(name="process_documents_task", bind=True)
def process_documents_task(self, file_paths):
    try:
        self.update_state(state="PROGRESS", meta={'message':'Loading and splitting documents'})
        result = rag_engine.process_documents(file_paths)
        
        # Cleanup temporary files
        for path in file_paths:
            file_to_remove = str(path)
            if os.path.exists(file_to_remove):
                try:
                    os.remove(file_to_remove)
                except Exception as e:
                    print(f"Cleanup error for {file_to_remove}: {e}")
                    
        return result
    except Exception as e:
        return f"Error in background task: {str(e)}"

