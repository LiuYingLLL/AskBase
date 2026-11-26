"""模型层公共工具

此前每个模型文件都各自定义了一份 `_now()`，14 份实现完全一样。
统一收敛到这里，避免时间格式在某个文件被改动后与其他表不一致。
"""

from datetime import datetime


def now_iso() -> str:
    """当前时间的 ISO 8601 字符串，用作 ORM 默认值"""
    return datetime.now().isoformat()
