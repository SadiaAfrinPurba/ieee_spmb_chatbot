import sys
import streamlit as st
import os
import toml
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import ollama

config = toml.load("config.toml")
urls = config["data_sources"]["urls"]

@st.cache_data
def load_and_process_documents(urls):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    return text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

if os.path.exists('db'):
    vectorstore = Chroma(persist_directory='db', embedding_function=embeddings)
else:
    with st.spinner("Loading and processing documents..."):
        documents = load_and_process_documents(urls)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory='db')
    
    vectorstore.persist()

retriever = vectorstore.as_retriever()


if 'messages' not in st.session_state:
    st.session_state.messages = []

def generate_response(question):
    start_time = time.perf_counter()
    st.session_state.messages.append({'role': 'user', 'content': question})
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    conversation_history = ''
    for msg in st.session_state.messages:
        conversation_history += f"{msg['role'].capitalize()}: {msg['content']}\n"

    formatted_prompt = f"{conversation_history}\nContext:\n{context}\n\nAssistant:"

    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': formatted_prompt}])
    assistant_reply = response['message']['content'].strip()

    st.session_state.messages.append({'role': 'assistant', 'content': assistant_reply})

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return assistant_reply, elapsed_time

st.title("IEEE SPMB Bot")


for msg in st.session_state.messages:
    if msg['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(msg['content'])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg['content'])


user_query = st.chat_input("Enter your message")
if user_query:
    with st.spinner("Generating response..."):
        response, elapsed_time = generate_response(user_query)

    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(f"*Response generated in {elapsed_time:.2f} seconds.*")
        

