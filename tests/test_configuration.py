"""配置模块基础校验。"""
from deep_research import configuration as cfg


def test_path_derivation():
    assert (cfg.ROOT / "src" / "deep_research").is_dir()
    assert cfg.DATA_DIR == cfg.ROOT / "data"
    assert cfg.REPORTS_DIR == cfg.ROOT / "reports"


def test_embedding_dims():
    assert cfg.EMBEDDING_DIMS == 1024  # qwen3-embedding:0.6b 原生维度，建库后不可变


def test_model_defaults_present():
    assert cfg.MODEL_RESEARCH and cfg.MODEL_SUMMARIZE
    assert cfg.EMBEDDING_MODEL
    assert cfg.RAG_TOP_K >= 1 and cfg.CHUNK_SIZE > cfg.CHUNK_OVERLAP
