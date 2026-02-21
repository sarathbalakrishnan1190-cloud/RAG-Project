import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
import requests
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import time
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT


class HuggingFaceRouterChat(BaseChatModel):
    repo_id: str
    token: str
    temperature: float = 0.01
    max_new_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "huggingface_router"

    def _generate(self, messages: list[BaseMessage], **kwargs):
        formatted = []
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else "system"
            formatted.append({"role": role, "content": str(m.content)})

        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "model": self.repo_id,
            "messages": formatted,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }

        
        for i in range(3):
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
            elif resp.status_code == 503:
                print(f"Model loading, retrying in 10s ({i+1}/3)...")
                time.sleep(10)
            else:
                raise Exception(f"HF Router error {resp.status_code}: {resp.text}")
        raise Exception("Model failed to load after retries.")

class RAGEngine:
    def __init__(self):
        load_dotenv()

        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documind_collection")

        
        print("Initializing HuggingFace Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        
        print("Initializing LLM via HuggingFaceRouterChat...")
        self.llm = HuggingFaceRouterChat(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
            max_new_tokens=512,
            temperature=0.01
        )

        
        try:
            print(f"Connecting to Qdrant at {self.qdrant_url}...")
            if self.qdrant_api_key:
                self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=30)
            else:
                self.qdrant_client = QdrantClient(url=self.qdrant_url, timeout=30)
            self.qdrant_client.get_collections()
            print("✅ Connected to Qdrant")
        except Exception:
            print(f"⚠️ Using local storage fallback")
            self.qdrant_client = QdrantClient(path="./qdrant_storage")

        
        collections = self.qdrant_client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            print(f"Creating collection {self.collection_name}...")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

    def process_documents(self, file_paths: List[str]) -> str:
        documents = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
            documents.extend(docs)

        if not documents:
            return "No documents to process."

        splits = splitter.split_documents(documents)
        self.vector_store.add_documents(documents=splits)
        return f"Processed {len(file_paths)} files into {len(splits)} chunks."

    def get_conversational_chain(self):
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the user question as standalone based on chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}\n\nContext:\n{context}")
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def query(self, question: str, chat_history: Optional[List] = None) -> str:
        if chat_history is None:
            chat_history = []

        chain = self.get_conversational_chain()
        response = chain.invoke({"input": question, "chat_history": chat_history})
        return response["answer"]
