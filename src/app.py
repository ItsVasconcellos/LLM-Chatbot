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

query_engine = index.as_query_engine(streaming=True,similarity_top_k=2,system_prompt=("Consider that you are an assistnat made"   "to respond about industrial safety standards accodring to the context provided. "
        "You shall give clear and concise answers based on the information provided in the context. "
        "If there is no match betweenthe context and the question, you should respond that the answer is not known."
        "Limit your response to 3 sentences. If really necessary, you can provide up to 5 sentences. "
        "If you need more information, ask the user for clarification. "
        "Always respond in English.\n\n"))

st.title("Welcome to the master of safety engineering!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask any question and the genius will answer it for you!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get Response from query engine
    response = query_engine.query(prompt)

    # Display chatbot response in chat message container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        streamed_response = ""
    
        for token in response.response_gen:
            streamed_response += token
            response_placeholder.write(streamed_response)  
            
    st.session_state.messages.append({"role": "assistant", "content": streamed_response})