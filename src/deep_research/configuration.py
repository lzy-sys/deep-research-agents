"""全局配置：.env 直读 + 派生路径。全项目唯一配置源。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
FAISS_DIR = DATA_DIR / "vector_store"  # FAISS 索引落盘目录（旧名 CHROMA_DIR 已废弃，引擎早已换 FAISS）
REPORTS_DIR = ROOT / "reports"
DB_PATH = DATA_DIR / "douban.sqlite"
MEMORY_DB_PATH = DATA_DIR / "memory.sqlite"  # 长期记忆库（结构化条目，替代早期 AGENTS.md 文件方案）

# ===== LLM：OpenCode Go 网关 =====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "opencode")  # opencode（网关）| ollama（本地）
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))  # 单次 LLM 调用超时秒数
LLM_THINKING = os.getenv("LLM_THINKING", "0") == "1"  # 1=开思考模式；0=关（多轮工具调用下思考内容回传会被上游 400 拒绝，且拖慢 5 倍）
LLM_EXTRA_BODY = None if LLM_THINKING else {"thinking": {"type": "disabled"}}  # 思考模式参数，llm.py 与探针脚本共用
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

# ===== SSE 事件流 =====
STREAM_IDLE_TIMEOUT = int(os.getenv("STREAM_IDLE_TIMEOUT", "660"))  # 无事件判超时秒数；api 判定与 webui 读超时共用同一常量

DOCS_LLMSTXT = "https://docs.langchain.com/llms.txt"
DOUBAN_CSV_URL = (
    "https://raw.githubusercontent.com/dengfuping/douban-movies-spider/main/"
    "data/douban_all_movies.csv"
)
