import os
import sys
from rag_engine import RAGEngine
from dotenv import load_dotenv

def test_rag_flow():
    load_dotenv()
    
    # 1. Initialize Engine
    print("Initializing RAG Engine...")
    try:
        engine = RAGEngine()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # 2. Check for sample files or skip upload test
    # Support multiple possible names for the test file
    possible_files = ["test_sample.pdf", "test_sample (2).pdf"]
    target_pdf = None
    
    for f in possible_files:
        path = os.path.join(os.getcwd(), f)
        if os.path.exists(path):
            target_pdf = path
            break
            
    if target_pdf:
        print(f"Processing test file: {target_pdf}")
        result = engine.process_documents([target_pdf])
        print(f"Result: {result}")
    else:
        print("⚠️ No test PDF found. Skipping upload test.")
        print("Expected one of: " + ", ".join(possible_files))

    # 3. Test Query
    test_query = "What is this document about?"
    print(f"\nTesting query: '{test_query}'")
    try:
        answer = engine.query(test_query)
        print(f"\nAnswer:\n{answer}")
    except Exception as e:
        print(f"Query failed: {e}")

    # 4. Streaming query (should not raise NotImplementedError)
    print("\nTesting streaming query")
    try:
        streamed = []
        for token in engine.stream_query(test_query):
            streamed.append(token)
        print("Streamed result:", "".join(streamed))
    except Exception as e:
        print(f"Streaming failed: {e}")

if __name__ == "__main__":
    test_rag_flow()
