FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# 先拷依赖清单再装依赖，利用层缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY webui ./webui
COPY scripts ./scripts
RUN mkdir -p reports data

ENV PYTHONUNBUFFERED=1
# venv 二进制直接进 PATH：运行时绕开 uv run 的依赖同步（容器内不联网补装）
ENV PATH="/app/.venv/bin:$PATH"
# 源码直导：uv 在 Linux 容器里对 Windows 路径来源的根项目安装不可靠，用 PYTHONPATH 兜底
ENV PYTHONPATH="/app/src"
EXPOSE 8000 8501
