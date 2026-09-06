"""pytest 全局配置：把项目根加入 sys.path，使 scripts.* 可导入。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
