# 📚 Documentation Support Agent  
A fast, reliable, and grounded **RAG-based documentation assistant** that lets you ingest PDFs, text files, URLs, or raw text — and then answers questions **only using the ingested documents**.

Built using:

✅ Streamlit  
✅ Semantic Chunking (LangChain SemanticChunker)  
✅ FAISS Vector Store  
✅ Sentence Transformers  
✅ Gemini 2.5 Flash  

Everything lives inside **one single file**:  
`doc_support_agent.py`

---

## 🚀 Features

### ✅ Multi-Source Ingestion  
- Upload **PDF/TXT**
- Paste raw text
- Enter **URLs** for automatic scraping

### ✅ Smart Semantic Chunking  
- Uses LangChain's **SemanticChunker** for meaningful chunk boundaries  


### ✅ Fast & Accurate Retrieval  
- Sentence Transformers embeddings  
- FAISS cosine similarity search  
- Top-k chunk ranking

### ✅ Grounded Answer Generation  
- Powered by **Gemini 2.5 Flash**  
- Strictly grounded to provided sources  
- Refuses to hallucinate  
- Returns clean citations

### ✅ Single-File Streamlit App  
- Simple for deployment  

---





