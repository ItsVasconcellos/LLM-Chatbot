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

llm = Gemini(model="models/gemini-1.5-pro", temperature=0.3, top_p=1, top_k=32, api_key=GOOGLE_API_KEY)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

if not os.path.exists(PERSIST_DIR):    
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

chat_engine = index.as_query_engine(similarity_top_k=2,system_prompt=("Consider that you are an assistnat made"   "to respond about industrial safety standards accodring to the context provided. "
        "You shall give clear and concise answers based on the information provided in the context. "
        "If there is no match betweenthe context and the question, you should respond that the answer is not known."
        "Limit your response to 3 sentences. If really necessary, you can provide up to 5 sentences. "
        "If you need more information, ask the user for clarification. "
        "Always respond in English.\n\n"))

st.title("Welcome to the master of engineering!")
st.text_input("Ask any question and the genius will answer it for you!", key="doubt")

if st.button("Ask the genius"):
    with st.spinner("Processing the data..."):
        response = chat_engine.query(st.session_state.doubt)
    st.write("Result: " + response.response)