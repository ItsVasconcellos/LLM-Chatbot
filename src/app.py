import streamlit as st 

import os
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_KEY")
PERSIST_DIR = "./tmp"

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("./data").load_data()

    llm  = Gemini(model="models/gemini-1.5-pro", temperature=0.3, top_p=1, top_k=32, api_key=GOOGLE_API_KEY)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)

    # global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

st.write("Bem vindo ao chatobt gênio da lâmpada!")
st.text_input("Tire sua dúvida aqui!", key="doubt")

if st.button("Enviar"):
    st.write("Pensando...")
    response = query_engine.query(st.session_state.doubt)
    st.write(response.response)