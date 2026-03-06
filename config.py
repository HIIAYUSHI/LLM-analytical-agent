import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

LLM_MODEL = "llama-3.3-70b-versatile"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "enterprise_data_context"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"