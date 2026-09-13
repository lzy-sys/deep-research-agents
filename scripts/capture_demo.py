"""Capture the Streamlit demo screen for the repository README."""
from pathlib import Path

from playwright.sync_api import sync_playwright

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "webui.png"


def _chromium_executable() -> str | None:
    system_paths = (
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    )
    for path in system_paths:
        if path.exists():
            return str(path)

    root = Path.home() / "AppData" / "Local" / "ms-playwright"
    patterns = (
        "chromium-*/chrome-win64/chrome.exe",
        "chromium-*/chrome-linux/chrome",
        "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    )
    matches = sorted(path for pattern in patterns for path in root.glob(pattern))
    return str(matches[-1]) if matches else None


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=_chromium_executable())
        page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1)
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.get_by_text("深度研究多智能体系统").wait_for(timeout=30_000)
        page.locator("textarea").fill("请比较 LangGraph 与 DeepAgents 的多智能体编排机制，并给出选型建议。")
        page.wait_for_timeout(3_000)
        page.screenshot(
            path=str(OUTPUT),
            clip={"x": 300, "y": 0, "width": 1140, "height": 960},
        )
        browser.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
