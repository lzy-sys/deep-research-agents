"""多轮 CLI 入口：新研究 / 追问 / 记忆管理 / 报告列表。

用法: uv run python -m src.deep_research.main
"""
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from deep_research import configuration as cfg
from deep_research.events import final_content, iter_tool_labels, new_thread_id
from deep_research.graph import build
from deep_research.memory import store

console = Console()

HELP = """[bold]命令[/]
  直接输入文字   新研究主题 / 针对当前研究的追问
  /new           开启新研究（新会话线程）
  /memory        查看长期记忆（数据库条目）
  /memory/reset  重置长期记忆
  /reports       列出已生成的报告
  /quit          退出"""


def show_reports() -> None:
    files = sorted(cfg.REPORTS_DIR.glob("*.md"))
    if not files:
        console.print("(还没有报告，输入研究主题生成一份)")
        return
    for f in files:
        console.print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


def run_turn(agent, config: dict, user_input: str) -> None:
    with console.status("[dim]supervisor 规划中…[/]"):
        for chunk in agent.stream({"messages": [("user", user_input)]}, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node != "model":
                    continue
                for label in iter_tool_labels(update):
                    console.print(f"  [magenta]•[/] {label}")
    content = final_content(agent.get_state(config).values.get("messages", []))
    if not content:
        console.print("[red](没有得到回复)[/]")
        return
    console.print(Panel(Markdown(content), title="回复", border_style="green"))


def main() -> int:
    console.print(Panel("深度研究多智能体系统\nsupervisor + web-researcher / rag-expert / sql-expert", title="Deep Research"))
    agent = build()
    config = {"configurable": {"thread_id": new_thread_id()}}

    if len(sys.argv) > 1:  # 非交互模式：命令行直接给研究主题
        run_turn(agent, config, " ".join(sys.argv[1:]))
        return 0

    console.print(f"[dim]thread: {config['configurable']['thread_id']} | 记忆: {cfg.MEMORY_DB_PATH.name}[/]\n{HELP}\n")
    while True:
        try:
            user = console.input("[bold cyan]你> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/new":
            config = {"configurable": {"thread_id": new_thread_id()}}
            console.print(f"[dim]新会话: {config['configurable']['thread_id']}[/]")
            continue
        if user == "/memory":
            entries = store.list_entries()
            body = store.format_memory() if entries else "（暂无记忆条目）"
            console.print(Panel(Markdown(body), title=f"长期记忆（{len(entries)} 条）"))
            continue
        if user == "/memory/reset":
            removed = store.reset_memory()
            agent = build()  # 重建使新记忆生效
            console.print(f"[dim]记忆已重置（清除 {removed} 条）[/]")
            continue
        if user == "/reports":
            show_reports()
            continue
        if user.startswith("/"):
            console.print(HELP)
            continue
        try:
            run_turn(agent, config, user)
        except KeyboardInterrupt:
            console.print("[yellow](本轮中断)[/]")
        except Exception as e:
            console.print(f"[red]出错了: {type(e).__name__}: {e}[/]\n[dim]检查 .env 配置后重试[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
