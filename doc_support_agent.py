import os
import warnings
import tempfile
from typing import List, Dict, Tuple

import numpy as np
import faiss
import PyPDF2
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")


from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    source: str
    chunk_id: int
    start_char: int
    end_char: int

class DocumentProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
       
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.chunker = SemanticChunker(self.embeddings)

    def process_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text += page_text + "\n"
        return text

    def process_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def process_url(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)

    def chunk_text(self, text: str, source: str) -> List[Chunk]:
        """
        Uses LangChain SemanticChunker to split input text.
        """
        semantic_chunks = self.chunker.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(semantic_chunks):
            chunks.append(Chunk(
                text=chunk_text.strip(),
                source=source,
                chunk_id=i,
                start_char=0,
                end_char=len(chunk_text)
            ))
        return chunks

class VectorStore:

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []

    def add_documents(self, chunks: List[Chunk]):
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.index or not self.chunks:
            return []
        query_embedding = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, min(k,len(self.chunks)))
        return [(self.chunks[idx], float(score)) for idx, score in zip(indices[0], distances[0])][:k]
    

    def reset(self):
        """Clear all stored documents and rebuild index."""
        self.index = None
        self.stored_chunks = []   


class AnswerGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": 1500,
            },
        )

    def generate_answer(self, query: str, chunks: List[Tuple[Chunk, float]]) -> str:
        context_parts = []
        for i, (chunk, score) in enumerate(chunks[:5], 1):
            context_parts.append(f"[Source {i}] (Relevance: {score:.1%}):\n{chunk.text}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are a documentation assistant that ONLY uses provided sources.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. Answer ONLY using information from the sources below
2. DO NOT use any external knowledge, even if you know the answer
3. DO NOT say "I can provide a general answer" or similar phrases
4. If sources mention the topic, extract and explain what they say
5. If the sources do NOT contain enough information:
- You MUST output exactly this sentence: "Insufficient information"
6. Always cite which source you're using: [Source 1], [Source 2], etc.
7. NEVER add code examples unless they appear in the sources
8. NEVER add explanations beyond what the sources state
9. You must ALWAYS return some text.

SOURCES:
{context}

QUESTION: {query}

ANSWER:
• A clear explanation of the answer using ONLY info from the sources.

ANSWER:"""

        try:
            response = self.model.generate_content(prompt)
            
            text = getattr(response, "text", "")
            
            if text.strip() == "":
                return "Insufficient information"

    
            if "insufficient" in text.lower():
                return "Insufficient information"

            return text.strip()
        except Exception as e:
            return f" Error generating answer: {str(e)}"


class Similar_source:
    def __init__(self, similarity_threshold: float = 0.30):
        self.similarity_threshold = similarity_threshold

    def format_sources_clean(self, chunks: List[Tuple[Chunk, float]]) -> List[Dict]:
        formatted_sources = []
        for i, (chunk, score) in enumerate(chunks[:5], 1):
            formatted_sources.append({
                "source_id": i,
                "similarity": f"{score:.2%}",
                "source": chunk.source,
                "excerpt": chunk.text,
            })
        return formatted_sources


class support_agent:
    def __init__(self, api_key: str):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator(api_key)
        self.similar_source = Similar_source()

    def _actual_source_name(self, source_type, source):
            if source_type == "pdf" or source_type == "txt":
                return os.path.basename(source)
            elif source_type == "url":
                return f"URL: {source}"
            elif source_type == "text":
                return "Text"
            return "Unknown Source"    

    def ingest_source(self, source_type: str, source: str) -> int:
        if source_type == "pdf":
            text = self.doc_processor.process_pdf(source)
        elif source_type == "txt":
            text = self.doc_processor.process_txt(source)
        elif source_type == "url":
            text = self.doc_processor.process_url(source)
        elif source_type == "text":
            text = source
        else:
            raise ValueError("Unsupported source type")

    
        actual_source = self._actual_source_name(source_type, source)
        chunks = self.doc_processor.chunk_text(text, actual_source) 

        self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def clear_documents(self):
        """Clear all ingested documents."""
        self.vector_store.clear()

    def query(self, question: str, k: int = 5)-> Dict:
        chunks = self.vector_store.search(question, k=k)
        answer = self.answer_generator.generate_answer(question, chunks)
        formatted = self.similar_source.format_sources_clean(chunks)
        return {"answer": answer, "sources": formatted}


st.set_page_config(page_title="📚 Documentation Support Agent", page_icon="🤖", layout="wide")

st.title("📚 Documentation Support Agent")

if "support_agent" not in st.session_state:
    st.session_state.support_agent = None
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "document_count" not in st.session_state: 
    st.session_state.document_count = 0    

api_key = st.text_input("🔑 Enter your Gemini API Key:", type="password")

if api_key and st.session_state.support_agent is None:
    try:
        st.session_state.support_agent = support_agent(api_key)
        st.success("✅ support_agent initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing support_agent: {str(e)}")

support_agent = st.session_state.support_agent
if support_agent is None:
    st.stop()

st.header(" Document Ingestion")
source_option = st.radio("Choose type:", ["Upload PDF/TXT", "Enter URL", "Paste Text"], horizontal=True)

if source_option == "Upload PDF/TXT":
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "txt"
        try:
            num_chunks = support_agent.ingest_source(file_type, temp_path)
            st.success(f"✅ Ingested {num_chunks} chunks from file.")
            st.session_state.ingested = True
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif source_option == "Enter URL":
    url = st.text_input("Enter a webpage URL:")
    if st.button("Ingest URL"):
        try:
            num_chunks = support_agent.ingest_source("url", url)
            st.success(f"✅ Ingested {num_chunks} chunks from URL.")
            st.session_state.ingested = True
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif source_option == "Paste Text":
    text = st.text_area("Paste your text here:")
    if st.button("Ingest Text"):
        if text.strip():
            try:
                num_chunks = support_agent.ingest_source("text", text)
                st.success(f"✅ Ingested {num_chunks} chunks from input text.")
                st.session_state.ingested = True
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please paste some text.")

if uploaded_file:
    with st.spinner("📄 Extracting text from document..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

    with st.spinner("🔍 Chunking document using Semantic Chunker..."):
        file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "txt"
        num_chunks = support_agent.ingest_source(file_type, temp_path)

    st.success(f"✅ Successfully processed {num_chunks} chunks!")
    st.session_state.ingested = True


st.header("💬 Ask a Question")
if not st.session_state.ingested:
    st.warning("Please ingest at least one document before querying.")
    st.stop()
 

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Generating answer..."):
        try:
            result = support_agent.query(query)
            st.subheader("✅ Answer")
            st.write(result["answer"])

            st.subheader("📚 Source Passages Used")
            for src in result.get("sources", []):
                with st.expander(f"Source {src['source_id']} (Similarity: {src['similarity']})"):
                    st.markdown(f"**From:** `{src['source']}`")  
                    st.write(src["excerpt"])
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
