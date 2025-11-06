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
- Better than fixed-size window chunking

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
- Easy to fork & modify  
- No backend server required  

---

## 📁 Project Structure

Documentation-Support-Agent/
│
├── doc_support_agent.py # ✅ Entire application (UI + backend)
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── Sample_documents/ # (Optional) for PDFs or screenshots


---

## ✅ Installation

### **2. Create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows

# OR

source .venv/bin/activate      # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

## ▶️ Run the App

Inside the project folder:

```bash
streamlit run doc_support_agent.py

User → Streamlit UI
     → DocumentProcessor (PDF/URL/Text parsing)
     → SemanticChunker → meaning-based chunking
     → SentenceTransformer embeddings
     → FAISS vector search
     → Gemini 2.5 Flash → grounded answer
     → UI displays answer + source evidence

## 🧪 Example Workflow

1. Enter your **Gemini API Key**
2. Upload a **PDF**, paste text, or enter a **URL**
3. Ask a question
4. View:
   - ✅ Generated Answer
   - ✅ Top relevant source passages


