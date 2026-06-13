"""
Configuration file for Document Knowledge Management System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== Basic Config ====================
PROJECT_NAME = "Document Knowledge Management System"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ==================== Database Config ====================
# Vector DB: FAISS or SQLite
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # "faiss" or "sqlite"
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/knowledge.db")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # Sentence-Transformers dimension

# ==================== LLM Config ====================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai", "huggingface", "local"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "THUDM/chatglm-6b")  # For local models

# ==================== NLP Models Config ====================
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Sentence-Transformers

# ==================== Document Processing Config ====================
PDF_EXTRACT_IMAGES = os.getenv("PDF_EXTRACT_IMAGES", "True").lower() == "true"
PDF_OCR_ENABLED = os.getenv("PDF_OCR_ENABLED", "False").lower() == "true"
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ==================== Knowledge Extraction Config ====================
NER_THRESHOLD = float(os.getenv("NER_THRESHOLD", "0.5"))
RELATION_THRESHOLD = float(os.getenv("RELATION_THRESHOLD", "0.5"))

# ==================== RAG Config ====================
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

# ==================== API Config ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

# Create necessary directories
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
