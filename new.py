#import libraries
from langchain_anthropic import ChatAnthropic
from langchain_core.prompt import PromptTemplate
from langchain_core.huggerface import HuggingFaceEmbeddings
from chromadb.config import Settings
import chromadb

from langchain_core.runnables import Runnable
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()
def AI_ANSWER():
    #SETUP ANTHROPIC
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.9, ANTHROPIC_API_KEY= os.getenv(api_key="ANTHROPIC_API_KEY"))


def prompt(context, question):
    #SETUP PROMPT
    prompt = PromptTemplate.from_template(

        "You are a helpful assistant that answers questions about the following text: {context}\n\nQuestion: {question}\nAnswer:"
    )
    user_prompt = prompt.content()
    return user_prompt
#Here is your prompt template. You can customize it as needed. The {context} will be replaced with the relevant information, and {question} will be replaced with the user's question. The assistant will then generate an answer based on the provided context and question.
my_prompt = prompt("Interview questions", " Give me a template of a regular typical interveiw for a generative AI engineer")

#CHUNK RESULT
chunks = []

def chunking_prompt(text, chunk_limit = 20):
    chunk_split = prompt(text).split("user_prompt")
    for i in range(0, len(chunk_split), chunk_limit):
        chunked =  chunk_split[i:i + chunk_limit]
        chunks.append(" ".join(chunked))
        st.write(f"My chunked list: {chunks}")
        for chunk in chunks:
            st.write(chunk)
    return chunks
knowledge_chunk = chunking_prompt(my_prompt)

def embedding(text):
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    emmbedded = model.core(prompt)
    return emmbedded

my_embedding = embedding(knowledge_chunk)




#EMBED RESULT AD STORE IN VECTOR DB
chroma_client = Chromadb.Client(Settings(persist_directory = "./chroma_store"))

collection = chroma_client.get_or_create_collection(name = "my_kb")

#compare vectors for similarity
for i , chunk in enumerate(knowledge_chunk):
    collection.add(
        documents = [chunk],
        ids = [f"chunk_{i}"],
        embeddings = [my_embedding]
    )   

#User query

query = st.text_input("Ask a question about the interview template", key="user_query")
query_embedding = embedding(query)

#Semantic search

results = collection.query(
    query_embeddings = [query_embedding],n_results = 2
)

top_chunks = results['documents'][0]





