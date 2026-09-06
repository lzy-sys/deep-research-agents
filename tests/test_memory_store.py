"""AGENTS.md 长期记忆读写测试（隔离到 tmp，不碰真实记忆）。"""
from deep_research.memory import store


def test_init_read_reset(tmp_path, monkeypatch):
    fake = tmp_path / "AGENTS.md"
    monkeypatch.setattr(store, "MEMORY_PATH", fake)

    content = store.read_memory()  # 不存在 → 自动初始化
    assert fake.exists()
    assert "用户偏好" in content and "暂无" in content

    fake.write_text("# 我被智能体更新过\n", encoding="utf-8")
    assert store.read_memory() == "# 我被智能体更新过\n"  # 读取的是最新内容

    store.reset_memory()
    assert "暂无" in store.read_memory()  # 重置恢复初始模板
