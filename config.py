from pathlib import Path

# URL parse
MAX_WORDS_PER_CHUNK = 100

# Vectorstore paths
DB_FOLDER = Path("vectorstore")
VECTOR_DB_PATH = DB_FOLDER / Path("index.faiss")
DOC_STORE_PATH = DB_FOLDER / Path("docs.pkl")

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval
LLM_NAME = "bitnet"
MODEL_PATH = "./models/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
CLI_PATH = "./models/BitNet/build/bin/llama-cli"
TOP_K_DOCS = 5
MAX_TOKENS = 256
N_THREADS = 4
CTX_SIZE = 4096
TEMPERATURE = 0.8

# Loggings
LOG_FOLDER = Path("logs")
DB_LOG_PATH = LOG_FOLDER / Path("build_vectorstore.log")
CHAT_LOG_PATH = LOG_FOLDER / Path("chat.log")
