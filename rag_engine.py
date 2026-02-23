import os
import requests
import logging
from typing import List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq # type: ignore
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_core.outputs import (
    ChatResult,
    ChatGeneration,
    ChatGenerationChunk,
)

from prompts import SYSTEM_PROMPT

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGEngine")


class OllamaChat(BaseChatModel):
    """Local Llama 3 via Ollama as last-resort fallback LLM."""
    model: str = "tinyllama"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.01
    max_new_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "ollama_chat"

    def _call_model(self, messages: list[BaseMessage], **kwargs) -> str:
        prompt_parts = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None) or "user"
            content = getattr(m, "content", str(m))
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n".join(prompt_parts)

        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                },
            }
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            return f"[Ollama error: {exc}]"

    def _stream(self, messages: list[BaseMessage], **kwargs):
        text = self._call_model(messages, **kwargs)
        yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    def _generate(self, messages: list[BaseMessage], **kwargs):
        res = ""
        for chunk in self._stream(messages, **kwargs):
            res += chunk.message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=res))]
        )


def _legacy_stream_query(question: str, chat_history: Optional[List] = None):
    if chat_history is None:
        chat_history = []

    engine = RAGEngine()
    response = engine.get_conversational_chain().invoke(
        {"input": question, "chat_history": chat_history}
    )

    full_text = response["answer"]
    for word in full_text.split():
        yield word + " "


class RAGEngine:
    def __init__(self):
        load_dotenv()

        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv(
            "QDRANT_COLLECTION_NAME",
            "documind_collection",
        )

        print("Initializing HuggingFace Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        ollama_model = os.getenv("OLLAMA_MODEL", "tinyllama")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        
        if groq_key:
            print("✅ Using Groq as primary LLM (llama-3.1-8b-instant) — free & fast!")
            from langchain_groq import ChatGroq # type: ignore
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.01,
                max_tokens=512,
                groq_api_key=groq_key,
            )
        elif openai_key:
            print("✅ Using OpenAI as LLM (gpt-4o-mini)")
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.01,
                max_tokens=512,
                openai_api_key=openai_key,
            )
        else:
            print(f"⚠️  No Groq/OpenAI key found. Falling back to Ollama (model={ollama_model})")
            print("   Make sure Ollama is running: `ollama serve`")
            self.llm = OllamaChat(
                model=ollama_model,
                base_url=ollama_url,
                temperature=0.01,
                max_new_tokens=512,
            )

        # Connect Qdrant
        try:
            print(f"Connecting to Qdrant at {self.qdrant_url}...")
            if self.qdrant_api_key:
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                )
            else:
                self.qdrant_client = QdrantClient(url=self.qdrant_url)

            self.qdrant_client.get_collections()
            print("✅ Connected to Qdrant")

        except Exception:
            print("⚠️  Using local Qdrant storage fallback")
            self.qdrant_client = QdrantClient(path="./qdrant_storage")

        collections = self.qdrant_client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def process_documents(self, file_paths: List[str]) -> str:
        documents = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        for file_path in file_paths:
            loader = UnstructuredPDFLoader(file_path)
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
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the user question as standalone."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            retriever,
            contextualize_prompt,
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}\n\nContext:\n{context}"),
        ])

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt,
        )

        return create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )

    def query(self, question: str, chat_history: Optional[List] = None) -> str:
        if chat_history is None:
            chat_history = []

        chain = self.get_conversational_chain()
        response = chain.invoke(
            {"input": question, "chat_history": chat_history}
        )
        return response["answer"]

    # -------- Streaming Query --------
    def stream_query(self, question: str, chat_history: Optional[List] = None):
        if chat_history is None:
            chat_history = []

        logger.info("stream_query called; question=%r history=%r", question, chat_history)
        chain = self.get_conversational_chain()

        yielded_any = False

        try:
            for chunk in chain.stream(
                {"input": question, "chat_history": chat_history}
            ):
                logger.debug("received chunk from chain.stream: %r", chunk)

                text_chunk = None

                if isinstance(chunk, dict):
                    if "answer" in chunk:
                        answer_chunk = chunk["answer"]
                        if hasattr(answer_chunk, "content"):
                            text_chunk = answer_chunk.content

                elif hasattr(chunk, "content"):
                    text_chunk = chunk.content
                elif hasattr(chunk, "text"):
                    text_chunk = chunk.text

                if text_chunk:
                    yielded_any = True
                    yield text_chunk

        except NotImplementedError as nie:
            logger.warning("chain.stream raised NotImplementedError, fallback: %s", nie)

        except Exception as err:
            logger.error("unexpected error in stream_query: %s", err, exc_info=True)
            raise

        if not yielded_any:
            try:
                response = chain.invoke(
                    {"input": question, "chat_history": chat_history}
                )
                full_text = response.get("answer", "")
                for word in full_text.split():
                    yield word + " "
            except Exception as e:
                logger.error("fallback invoke failed: %s", e, exc_info=True)
                raise