"""语料垃圾块过滤测试。"""
from scripts.ingest import is_junk


def test_nav_json_signature():
    assert is_junk('{"title":"Event streaming","href":"/oss/javascript/deepagents/streaming"}')


def test_too_short():
    assert is_junk("太短")


def test_high_punct_density():
    assert is_junk('{"a":"b","c":"d"}' * 20)


def test_normal_chinese():
    assert not is_junk("LangGraph 通过 StateGraph 定义节点和边，支持条件路由、并行派发与检查点持久化。" * 3)


def test_normal_code():
    code = 'def add(a, b):\n    return a + b\n\ngraph.add_node("add", add)  # 注册节点\n'
    assert not is_junk(code * 3)


def test_normal_english():
    assert not is_junk("LangGraph provides durable execution for long-running stateful agents." * 3)
