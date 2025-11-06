# 📚 Documentation Support Agent

A production-ready RAG (Retrieval-Augmented Generation) system with **zero-hallucination guarantees** for document-based question answering.

## 🎯 Features

- ✅ **Multi-source ingestion**: PDF, TXT, URLs, and raw text
- ✅ **Semantic search**: Uses sentence-transformers embeddings
- ✅ **FAISS vector store**: Fast similarity search
- ✅ **LLM-powered answers**: Gemini 1.5 Flash for generation
- ✅ **Hallucination prevention**: Strict source-based responses
- ✅ **Source highlighting**: Shows exact passages used
- ✅ **Two interfaces**: CLI and Streamlit web UI

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install sentence-transformers faiss-cpu PyPDF2 beautifulsoup4 requests google-generativeai streamlit numpy
```

### API Key Setup

Get a free Gemini API key from: https://ai.google.dev/

```bash
# Option 1: Environment variable
export GEMINI_API_KEY="your_key_here"

# Option 2: Enter when prompted
```

### Run CLI Version

```bash
python documentation_agent.py
```

### Run Web UI (Streamlit)

```bash
streamlit run streamlit_app.py
```

## 📖 Usage

### CLI Interface

1. **Ingestion Phase**
   - Choose source type (PDF, TXT, URL, or text)
   - Upload/paste your documentation
   - Repeat for multiple sources

2. **Query Phase**
   - Ask questions about your documents
   - Get answers with source citations
   - System refuses if information is insufficient

### Web Interface

1. Enter Gemini API key in sidebar
2. Upload documents via sidebar
3. Ask questions in the chat interface
4. View source passages in expandable sections

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User Query                         │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│        Semantic Search (FAISS + Embeddings)         │
│  • Encodes query to 384-dim vector                  │
│  • Cosine similarity with all chunks                │
│  • Returns top-15 most relevant chunks              │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│           LLM Generation (Gemini)                   │
│  • Strict prompt: "ONLY use sources"                │
│  • Temperature 0.1 (deterministic)                  │
│  • Refuses if sources insufficient                  │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│          Answer + Source Citations                  │
└─────────────────────────────────────────────────────┘
```

## 🔍 How It Works

### 1. Document Processing
- Extracts text from PDFs, TXT files, or URLs
- Splits into chunks (1000 chars with 200 overlap)
- Preserves section headers for better context

### 2. Embedding & Indexing
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Generates 384-dimensional embeddings
- Stores in FAISS for fast retrieval

### 3. Semantic Search
- Encodes user query to embedding
- Computes cosine similarity with all chunks
- Returns top-k most relevant passages

### 4. Answer Generation
- Sends query + top chunks to Gemini
- Strict prompt prevents hallucination
- LLM synthesizes answer from sources only

### 5. Hallucination Prevention
- **Layer 1**: Low temperature (0.1) for deterministic output
- **Layer 2**: Explicit prompt instructions
- **Layer 3**: Source citation requirement

## 🛡️ Hallucination Guardrails

The system employs multiple strategies to prevent hallucination:

1. **Strict Prompting**
   ```
   "Answer ONLY using information from sources"
   "DO NOT use external knowledge"
   "If insufficient, say so clearly"
   ```

2. **Low Temperature** (0.1)
   - Reduces creativity/randomness
   - Ensures consistent, grounded responses

3. **Source Citation**
   - Forces LLM to reference [Source 1], [Source 2]
   - Makes grounding transparent

4. **Similarity Threshold**
   - Only proceeds if relevance > 25%
   - Refuses automatically if no good matches

## 📊 Example Interactions

### ✅ Good Answer (Sources Available)
```
Q: What is list comprehension in Python?
A: List comprehensions provide a concise way to create lists [Source 1]. 
   A list comprehension consists of brackets containing an expression 
   followed by a for clause [Source 2].

Sources: 
  [Source 1] - 62.76% similarity
  [Source 2] - 55.29% similarity
```

### ✅ Correct Refusal (Sources Insufficient)
```
Q: How do I deploy to AWS?
A: The provided sources do not contain sufficient information to answer 
   this question about AWS deployment.
```

## 🧪 Testing

Test with different scenarios:

1. **Clear answers**: Questions directly covered in docs
2. **Partial info**: Questions partially covered
3. **No info**: Questions not in docs (should refuse)
4. **Multi-doc**: Questions spanning multiple sources

## 📁 Project Structure

```
.
├── documentation_agent.py    # Main CLI script
├── streamlit_app.py          # Web UI
├── README.md                 # This file
└── requirements.txt          # Dependencies
```

## 🔧 Configuration

### Chunking Parameters
```python
chunk_size = 1000      # Characters per chunk
chunk_overlap = 200    # Overlap between chunks
```

### Retrieval Parameters
```python
k = 15                 # Number of chunks to retrieve
similarity_threshold = 0.25  # Minimum relevance score
```

### LLM Parameters
```python
temperature = 0.1      # Low for consistency
max_tokens = 1500      # Maximum response length
top_p = 0.9           # Nucleus sampling
```

## 📝 Requirements

```txt
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
PyPDF2>=3.0.0
beautifulsoup4>=4.12.0
requests>=2.31.0
google-generativeai>=0.3.0
streamlit>=1.28.0
numpy>=1.24.0
```

## 🎓 Technical Highlights

1. **Pure Semantic Search**
   - No keyword matching (as per requirements)
   - Uses transformer-based embeddings
   - Demonstrates understanding of modern RAG

2. **Modular Architecture**
   - Clean separation of concerns
   - Easy to extend/modify
   - Production-ready structure

3. **Robust Error Handling**
   - Graceful failures
   - Informative error messages
   - Input validation

4. **Source Transparency**
   - Always shows which passages were used
   - Similarity scores visible
   - Builds trust in system

## 🚨 Known Limitations

1. **Context Window**: Limited to top-15 chunks
2. **PDF Quality**: Depends on text extraction quality
3. **Language**: Optimized for English
4. **Streaming**: Answers not streamed (could be added)

## 🔮 Future Enhancements

- [ ] Add support for DOCX, PPTX
- [ ] Implement conversation history
- [ ] Add re-ranking stage (e.g., cross-encoder)
- [ ] Support for multiple languages
- [ ] Add caching for faster repeated queries
- [ ] Implement query expansion

## 📄 License

This project was created as part of a technical interview assessment.

## 👤 Author

Created for Documentation Support Agent position interview.

---

**Note**: This is a demonstration project showcasing RAG architecture, embedding-based retrieval, and hallucination prevention techniques. The system prioritizes accuracy over coverage - it will refuse to answer rather than make up information.