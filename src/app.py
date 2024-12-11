import streamlit as st 
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="llama2", request_timeout=50.0)

# Correct the argument for HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = SimpleDirectoryReader("./data").load_data()

vector_index = VectorStoreIndex.from_documents(documents)

vector_index.as_query_engine()

st.write("Bem vindo ao chatobt gênio da lâmpada!")
st.text_input("Tire sua dúvida aqui!", key="doubt")

if st.button("Enviar"):
    st.write("Pensando...")
#     # response = Settings.llm.query(st.session_state.doubt)