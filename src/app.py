import streamlit as st 
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

Settings.llm = Ollama(model="llama2", request_timeout=50.0)

st.write("Bem vindo ao chatobt gênio da lâmpada!")
st.text_input("Tire sua dúvida aqui!", key="doubt")

if st.button("Enviar"):
    st.write("Pensando...")
    # response = Settings.llm.query(st.session_state.doubt)