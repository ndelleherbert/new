# IMPORTS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import os
import streamlit as st

# LOAD ENV
load_dotenv()

# LLM SETUP
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0.7,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# PROMPT FUNCTION
def build_prompt(context, question):
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    return prompt.format(context=context, question=question)

# CHUNKING FUNCTION
def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

# EMBEDDING FUNCTION
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedder.embed_documents(texts)

def embed_query(text):
    return embedder.embed_query(text)

# SAMPLE DATA
text_data = "This is a typical interview for a generative AI engineer including system design, NLP, LLMs, and deployment."

chunks = chunk_text(text_data)

# CHROMA SETUP
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_store"))
collection = chroma_client.get_or_create_collection(name="my_kb")

# STORE EMBEDDINGS
embeddings = embed_texts(chunks)

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"],
        embeddings=[embeddings[i]]
    )

# STREAMLIT UI
query = st.text_input("Ask a question")

if query:
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    top_chunks = results["documents"][0]
    context = " ".join(top_chunks)

    final_prompt = build_prompt(context, query)

    response = llm.invoke(final_prompt)

    st.write(response.content)