"""一次性验证脚本：用真实 PDF 打通 MinerU 在线解析

用法（backend 目录下）：
    venv\\Scripts\\python.exe scripts/verify_mineru.py <pdf路径>
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag.mineru_loader import is_mineru_configured, parse_pdf_with_mineru


async def main(path: str) -> int:
    if not is_mineru_configured():
        print("MinerU 未配置：检查 MINERU_ENABLED 与 MINERU_API_TOKEN")
        return 1

    print(f"开始解析: {os.path.basename(path)}")
    try:
        markdown = await parse_pdf_with_mineru(path)
    except Exception as e:
        print(f"解析失败: {e}")
        return 1

    print(f"解析成功，Markdown 长度: {len(markdown)} 字符")
    # MinerU 对复杂表格输出 HTML（能表达 rowspan/colspan），简单表格才用管道符
    print(f"含 HTML 表格: {'<table' in markdown}")
    print(f"含管道表格: {'|' in markdown}")
    print(f"含 Markdown 标题: {'#' in markdown[:200]}")
    print("---- 前 600 字 ----")
    print(markdown[:600])
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/verify_mineru.py <pdf路径>")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
