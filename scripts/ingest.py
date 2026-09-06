"""抓取 LangChain 官方文档入库：llms.txt → 下载 .md → 切块 → Chroma。

用法:
    uv run python scripts/ingest.py            # 已建库则跳过
    uv run python scripts/ingest.py --refresh  # 删库重建
"""
import hashlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from deep_research import configuration as cfg  # noqa: E402
from deep_research.tools.retrieval import get_embeddings, get_vectorstore, retrieve_docs, save_vectorstore  # noqa: E402

SECTION_RE = re.compile(r"\((https://docs\.langchain\.com/oss/python/[a-z0-9-]+/llms\.txt)\)")
BIGMODEL_LLMSTXT = "https://docs.bigmodel.cn/llms.txt"


def is_junk(text: str) -> bool:
    """JSON 导航垃圾块 / 过短块过滤。"""
    if len(text.strip()) < 40:
        return True
    if '"href"' in text and '"title"' in text:  # 站点导航 JSON 签名
        return True
    punct = sum(text.count(c) for c in '{}[]"')
    return punct / len(text) > 0.15


def fetch(url: str, retries: int = 2) -> str:
    for i in range(retries + 1):
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            return r.text
        except Exception:
            if i == retries:
                raise
            time.sleep(1 + i)


def collect_page_urls() -> dict[str, list[str]]:
    """返回 {板块: [页面 .md url]}：LangChain 三板块（英文）+ 智谱 bigmodel（中文）。"""
    main = fetch(cfg.DOCS_LLMSTXT)
    indexes = [
        u for u in SECTION_RE.findall(main)
        if re.search(r"/oss/python/(langchain|langgraph|deep)", u)
    ]
    pages: dict[str, list[str]] = {}
    for idx_url in indexes:
        sec = idx_url.split("/oss/python/")[1].split("/")[0]
        urls = re.findall(r"\((https://docs\.langchain\.com/oss/python/[^)]+\.md)\)", fetch(idx_url))
        urls = [u for u in urls if "changelog" not in u]  # 发布日志是检索黑洞，不入库
        pages[sec] = sorted(set(urls))
        print(f"  {sec}: {len(urls)} 页")

    bigmodel = sorted(set(re.findall(r"\((https://docs\.bigmodel\.cn/cn/[^)]+\.md)\)", fetch(BIGMODEL_LLMSTXT))))
    pages["bigmodel-zh"] = bigmodel
    print(f"  bigmodel-zh: {len(bigmodel)} 页")
    return pages


def download_all(pages: dict[str, list[str]]) -> list[Document]:
    def dl(item: tuple[str, str]) -> Document | None:
        sec, url = item
        try:
            slug = re.sub(r"^https?://[^/]+/", "", url).replace("/", "__")  # 通用 slug，跨站不冲突
            path = cfg.KB_DIR / sec / slug
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():  # 已下载过则跳过
                path.write_text(fetch(url), encoding="utf-8")
            content = path.read_text(encoding="utf-8")
            title = re.search(r"^#\s+(.+)$", content, re.M)
            return Document(
                page_content=content,
                metadata={"source": f"{sec}/{slug}", "title": title.group(1).strip() if title else slug},
            )
        except Exception as e:
            print(f"  [跳过] {url}: {e}")
            return None

    flat = [(sec, u) for sec, urls in pages.items() for u in urls]
    print(f"开始下载 {len(flat)} 页 ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        docs = [d for d in pool.map(dl, flat) if d]
    print(f"下载完成: {len(docs)}/{len(flat)} 页")
    return docs


def build_index(docs: list[Document]) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
    chunks = [c for c in splitter.split_documents(docs) if not is_junk(c.page_content)]
    seen: set[str] = set()
    uniq = []
    for c in chunks:
        digest = hashlib.md5(f"{c.metadata['source']}#{c.page_content}".encode()).hexdigest()
        if digest in seen:  # 重复内容的短块，批内去重
            continue
        seen.add(digest)
        c.metadata["chunk_hash"] = digest
        uniq.append(c)
    chunks = uniq
    print(f"切块: {len(chunks)} 块，开始 embedding（本地 Ollama，预计十几分钟）...")

    embeddings = get_embeddings()
    vs: FAISS | None = None
    batch = 64
    for i in range(0, len(chunks), batch):
        part_chunks = chunks[i : i + batch]
        for attempt in range(3):  # Ollama 偶发 502，退避重试
            try:
                part = FAISS.from_documents(part_chunks, embeddings)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  [重试 {attempt + 1}/3] 批次 {i}: {e}", flush=True)
                time.sleep(15 * (attempt + 1))
        if vs is None:
            vs = part
        else:
            vs.merge_from(part)
        print(f"  {min(i + batch, len(chunks))}/{len(chunks)}", flush=True)

    save_vectorstore(vs, cfg.CHROMA_DIR)
    print(f"入库完成，共 {vs.index.ntotal} 块 -> {cfg.CHROMA_DIR}")


def main() -> None:
    if (cfg.CHROMA_DIR / "index.faiss.bin").exists() and "--refresh" not in sys.argv:
        print(f"知识库已存在（{cfg.CHROMA_DIR}），跳过。重建请加 --refresh")
    else:
        docs = download_all(collect_page_urls())
        build_index(docs)

    print("\n===== 检索冒烟测试（从磁盘重新加载，模拟新进程）=====")
    get_vectorstore.cache_clear()
    print(retrieve_docs.invoke({"question": "What is create_deep_agent and how do subagents work?"})[:600])


if __name__ == "__main__":
    main()
