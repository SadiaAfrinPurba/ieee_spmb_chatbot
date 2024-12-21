import sys

import streamlit as st
import os
import toml
import time
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

import torch
from huggingface_hub import hf_hub_download

config = toml.load("config.toml")
urls = config["data_sources"]["urls"]
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = config["model"]["path"]

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            hf_hub_download(repo_id="TheBloke/Llama-2-7B-Chat-GGUF", filename="llama-2-7b-chat.Q4_K_M.gguf", local_dir="models/llama-2-7b-optimized")
    model = Llama(
        model_path=model_path, 
        n_ctx=2048,  
        n_threads=8,  
        n_batch=512,
        n_gpu_layers=32,  
        offload_kqv=True,  
        vocab_only=False, 
        use_mmap=True,    
        use_mlock=False,
        flash_attn=True   
    )
    return model

model = load_model()

@st.cache_data
def load_and_process_documents(urls):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': device},
    encode_kwargs={'device': device}
)

if os.path.exists('db'):
    vectorstore = Chroma(
        persist_directory='db',
        embedding_function=embeddings
    )
else:
    with st.spinner("Loading and processing documents..."):
        documents = load_and_process_documents(urls)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory='db')
    vectorstore.persist()

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

if 'messages' not in st.session_state:
    st.session_state.messages = []

def generate_response(question, max_tokens=100): #TODO: max_token will go to config file
    start_time = time.perf_counter()
    st.session_state.messages.append({'role': 'user', 'content': question})
    
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs[:2])
    
    conversation_history = ''
    for msg in st.session_state.messages[-2:]:
        conversation_history += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    prompt = f"""Instruct: Provide concise response.
    Context: {context}
    History: {conversation_history}
    Question: {question}
    Answer:"""

    
    response = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=0.9,
        repeat_penalty=1.1,
        echo=False
    )
    
    generated_text = response['choices'][0]['text']
    st.session_state.messages.append({'role': 'assistant', 'content': generated_text})
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return generated_text, elapsed_time

st.title("IEEE SPMB")

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

user_query = st.chat_input("Enter your message")
# with st.chat_message('user'):
#     st.markdown(user_query)
    
if user_query:
    # st.session_state.messages.append({'role': 'user', 'content': user_query})
    
    with st.chat_message('user'):
        st.markdown(user_query)
    
    with st.spinner("Generating response..."):
        try:
            response, elapsed_time = generate_response(user_query)
            with st.chat_message("assistant"):
                st.markdown(response)
                st.markdown(f"*Response generated in {elapsed_time:.2f} seconds.*")
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")