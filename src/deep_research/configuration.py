"""全局配置：.env 直读 + 派生路径。全项目唯一配置源。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
CHROMA_DIR = DATA_DIR / "vector_store"
REPORTS_DIR = ROOT / "reports"
DB_PATH = DATA_DIR / "douban.sqlite"

# ===== LLM：OpenCode Go 网关 =====
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "")
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
MODEL_RESEARCH = os.getenv("MODEL_RESEARCH", "deepseek-v4-flash")
MODEL_SUMMARIZE = os.getenv("MODEL_SUMMARIZE", MODEL_RESEARCH)
MODEL_REPORT = os.getenv("MODEL_REPORT", MODEL_RESEARCH)

# ===== 搜索 =====
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily")

# ===== Embedding：本地 Ollama =====
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_DIMS = 1024  # qwen3-embedding:0.6b 原生维度；建库后不可再改

# ===== RAG =====
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RAG_TOP_K = 5

DOCS_LLMSTXT = "https://docs.langchain.com/llms.txt"
DOUBAN_CSV_URL = (
    "https://raw.githubusercontent.com/dengfuping/douban-movies-spider/main/"
    "data/douban_all_movies.csv"
)
