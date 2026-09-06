"""长期记忆 SQLite 存储测试（DB 路径与旧文件路径均隔离到 tmp）。"""
import pytest

from deep_research.memory import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "memory.sqlite")
    monkeypatch.setattr(store, "LEGACY_MD_PATH", tmp_path / "AGENTS.md")  # 默认指向不存在的路径
    return store


def test_add_list_format_reset(db):
    pref_id, _ = db.add_entry("偏好", "模型只用 deepseek")
    db.add_entry("历史研究结论", "LangGraph 是低层编排运行时")

    entries = db.list_entries()
    assert len(entries) == 2
    assert entries[0]["id"] == pref_id and entries[0]["category"] == "偏好"  # 偏好排前

    text = db.format_memory()
    assert f"- [{pref_id}] 模型只用 deepseek" in text
    assert "## 历史研究结论" in text

    assert db.reset_memory() == 2
    assert db.list_entries() == []
    assert "（暂无长期记忆）" in db.format_memory()


def test_add_rejects_bad_input(db):
    with pytest.raises(ValueError):
        db.add_entry("杂项", "x")  # 非法类别
    with pytest.raises(ValueError):
        db.add_entry("偏好", "   ")  # 空内容


def test_delete_entry(db):
    entry_id, _ = db.add_entry("偏好", "x")
    assert db.delete_entry(entry_id) is True
    assert db.delete_entry(entry_id) is False  # 二次删除报不存在


def test_capacity_evicts_oldest_conclusion_only(db, monkeypatch):
    monkeypatch.setattr(store, "CAPACITY", 3)
    db.add_entry("历史研究结论", "旧结论 1")
    db.add_entry("历史研究结论", "旧结论 2")
    keep_id, _ = db.add_entry("偏好", "永不淘汰的偏好")
    _, note = db.add_entry("历史研究结论", "新结论")

    assert "淘汰" in note
    remaining = {e["id"]: e["content"] for e in db.list_entries()}
    assert keep_id in remaining and remaining[keep_id] == "永不淘汰的偏好"
    assert "旧结论 1" not in remaining.values()  # 最旧的结论被淘汰
    assert len(remaining) == 3


def test_capacity_full_without_conclusion_raises(db, monkeypatch):
    monkeypatch.setattr(store, "CAPACITY", 1)
    db.add_entry("偏好", "唯一一条偏好")
    with pytest.raises(RuntimeError):
        db.add_entry("偏好", "再放不下")


def test_migration_from_legacy_agents_md(db, monkeypatch, tmp_path):
    legacy = tmp_path / "AGENTS.md"
    legacy.write_text(
        "# 标题行不导入\n\n"
        "引导段落也不导入。\n\n"
        "## 用户偏好\n\n"
        "- 语言用中文\n"
        "（暂无，会话中得知后自动更新）\n\n"
        "## 历史研究结论\n\n"
        "- 结论 A\n"
        "- 结论 B\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "LEGACY_MD_PATH", legacy)

    entries = {e["content"] for e in db.list_entries()}  # 首次访问触发建库 + 迁移
    assert entries == {"语言用中文", "结论 A", "结论 B"}

    # 迁移只跑一次：清空后不得从旧文件重复导入
    db.reset_memory()
    assert db.list_entries() == []
