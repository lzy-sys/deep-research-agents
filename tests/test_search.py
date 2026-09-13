"""搜索工具的不可信内容边界测试。"""
from deep_research.tools import search


class _FakeSearch:
    def invoke(self, payload: dict) -> dict:
        return {"query": payload["query"], "results": [{"url": "https://example.com"}]}


def test_web_search_wraps_untrusted_content(monkeypatch):
    monkeypatch.setattr(search, "_basic_search", lambda: _FakeSearch())
    out = search.web_search.invoke({"query": "LangGraph"})
    assert out.startswith("<untrusted_web_content>")
    assert "https://example.com" in out
    assert out.endswith("</untrusted_web_content>")
