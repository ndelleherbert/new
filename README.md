\# 🔍 RAG Q\&A Assistant



A Streamlit application that combines \*\*Retrieval-Augmented Generation (RAG)\*\* with Claude (Anthropic), ChromaDB, and HuggingFace embeddings to answer questions grounded in your own knowledge base.



\---



\## Features



\- 💬 \*\*Chat\*\* — conversational Q\&A with retrieved context shown per response

\- 📄 \*\*Knowledge Base Manager\*\* — paste, edit, rebuild, and browse your document chunks

\- 📊 \*\*Chat Analytics\*\* — response time metrics, query log, and charts

\- 🧭 \*\*Vector Explorer\*\* — cosine similarity search and 2D PCA embedding projection

\- ⚙️ \*\*Settings\*\* — configure model, temperature, chunk size, retrieval count, and embedder



\---



\## Tech Stack



| Layer | Tool |

|---|---|

| LLM | Claude via `langchain-anthropic` |

| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |

| Vector store | ChromaDB (persistent local store) |

| UI | Streamlit |

| Env management | python-dotenv |



\---



\## Getting Started



\### 1. Clone the repo



```bash

git clone https://github.com/your-username/your-repo.git

cd your-repo

```



\### 2. Install dependencies



```bash

pip install -r requirements.txt

```



\### 3. Set up your API key



Create a `.env` file in the project root:



```

ANTHROPIC\_API\_KEY=your\_api\_key\_here

```



> Get your key at \[console.anthropic.com](https://console.anthropic.com)



\### 4. Run the app



```bash

python -m streamlit run new.py

```



\---



\## Requirements



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



Or install all at once:



```bash

pip install -r requirements.txt

```



\---



\## Project Structure



```

.

├── new.py              # Main Streamlit app

├── .env                # API key (not committed)

├── .gitignore          # Excludes .env and chroma\_store

├── requirements.txt    # Python dependencies

├── chroma\_store/       # Local ChromaDB vector store (auto-created)

└── README.md

```



\---



\## Usage



1\. Open the \*\*Knowledge Base\*\* dashboard and paste your document

2\. Click \*\*Save \& Rebuild\*\* to chunk and embed the text

3\. Switch to \*\*Chat\*\* and ask questions

4\. Explore \*\*Analytics\*\* and \*\*Vector Explorer\*\* for insights into your pipeline



\---



\## .gitignore



Make sure your `.gitignore` includes:



```

.env

chroma\_store/

\_\_pycache\_\_/

\*.pyc

```



\---



\## License



MIT

