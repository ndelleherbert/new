<div align="center">

# 🔍 RAG Q&A Assistant

**A multi-dashboard Streamlit app for document Q&A powered by Claude, ChromaDB, and HuggingFace.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-D97757?style=flat&logo=anthropic&logoColor=white)](https://anthropic.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange?style=flat)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)

</div>

---

## 📸 Dashboards

| Dashboard | Description |
|---|---|
| 💬 **Chat** | Conversational Q&A with retrieved context shown per response |
| 📄 **Knowledge Base** | Paste, edit, rebuild, and browse your document chunks |
| 📊 **Analytics** | Response time metrics, query log, and bar charts |
| 🧭 **Vector Explorer** | Cosine similarity search and 2D PCA embedding projection |
| ⚙️ **Settings** | Configure model, temperature, chunk size, and retrieval count |

---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| 🤖 LLM | `claude-sonnet-4-6` via `langchain-anthropic` |
| 🧠 Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| 🗄️ Vector Store | ChromaDB — persistent local store |
| 🖥️ UI | Streamlit |
| 🔐 Env | python-dotenv |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

> 🔑 Get your key at [console.anthropic.com](https://console.anthropic.com)

### 4. Run the app

```bash
python -m streamlit run new.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Requirements

```
langchain-anthropic
langchain-core
langchain-community
chromadb
sentence-transformers
streamlit
python-dotenv
scikit-learn
pandas
numpy
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
📁 your-repo/
├── 📄 new.py               # Main Streamlit app
├── 📄 requirements.txt     # Python dependencies
├── 📄 README.md            # You are here
├── 📄 .env                 # API key — never commit this
├── 📄 .gitignore
└── 📁 chroma_store/        # Local vector DB (auto-created on first run)
```

---

## 🧭 How It Works

```
Your document
     │
     ▼
 chunk_text()         Split into 100-word chunks
     │
     ▼
 HuggingFace          Convert chunks → embedding vectors
 Embedder
     │
     ▼
 ChromaDB             Store vectors in local persistent DB
     │
     ▼
 User Query           Embed query → similarity search → top N chunks
     │
     ▼
 Claude LLM           chunks + query → grounded answer
     │
     ▼
 Streamlit UI         Display answer + retrieved context
```

---

## 🛡️ .gitignore

Ensure these are excluded from version control:

```gitignore
.env
chroma_store/
__pycache__/
*.pyc
*.pyo
.DS_Store
```

---

## 📋 Usage Guide

1. Go to **📄 Knowledge Base** → paste your document → click **Save & Rebuild**
2. Switch to **💬 Chat** → ask questions about your document
3. Check **📊 Analytics** to monitor response times and query history
4. Use **🧭 Vector Explorer** to visualize how your chunks relate to each other
5. Tune **⚙️ Settings** to change the model, temperature, or chunk size

---

## 📄 License

MIT © 2025 — free to use, modify, and distribute.