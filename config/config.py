import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCq_PkkOA-UMmkOREAUmkv0Pg_2t_PZLbQ")

# Default allowed file types
ALLOWED_FILE_TYPES = [
    # Documents
    "pdf", "txt", "docx", "pptx", 
    # Images
    "png", "jpg", "jpeg", 
    # Audio
    "mp3", "wav", 
    # Video
    "mp4"
]

# File size limits (in bytes)
MAX_FILE_SIZE = {
    "default": 10 * 1024 * 1024,  # 10 MB default
    "pdf": 20 * 1024 * 1024,      # 20 MB for PDFs
    "image": 5 * 1024 * 1024,     # 5 MB for images
    "audio": 15 * 1024 * 1024,    # 15 MB for audio
    "video": 50 * 1024 * 1024     # 50 MB for video
}

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Temporary directory for file processing
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# CrewAI configuration settings
CREWAI_CONFIG = {
    "verbose": True,
    "memory": True
}

# Gemini model settings
GEMINI_CONFIG = {
    "text_model": "gemini-1.5-pro",
    "multimodal_model": "gemini-1.5-pro-vision",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_output_tokens": 2048
}

# RAG configuration
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "vectorstore_type": "faiss"  # Options: faiss, chroma, etc.
} 