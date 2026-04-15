"""Configuration module for RAG Document Q&A system."""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_SEED = 42
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
API_HOST = "0.0.0.0"
API_PORT = 8001
for d in [DATA_DIR]: d.mkdir(parents=True, exist_ok=True)
