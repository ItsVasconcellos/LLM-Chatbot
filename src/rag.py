from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

documents = SimpleDirectoryReader(
    input_files=["./data/Engineering-workshop-health-and-safety-guidelines-catalog.pdf"]
).load_data()


llm  = Gemini(model="models/gemini-1.5-pro", temperature=0.3, top_p=1, top_k=32)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)

# global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine()

response = query_engine.query(
    "What is the easiest way to communicate with someone who does not know your language?"
)

print(str(response))