"""批量测试：MinerU 解析 + 父子块切分的结构质量

只做解析与切分统计，不写库、不调 embedding。

用法（backend 目录下）：
    venv\\Scripts\\python.exe scripts/test_batch_parse.py <pdf1> <pdf2> ...
"""

import asyncio
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag.mineru_loader import parse_pdf_with_mineru  # noqa: E402
from app.rag.parent_child import split_with_parents  # noqa: E402


_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _fmt(n: int) -> str:
    return f"{n:,}"


async def test_one(path: str) -> None:
    name = os.path.basename(path)
    print(f"\n{'=' * 70}")
    print(f"文件: {name}")
    print(f"大小: {os.path.getsize(path) / 1024:.1f} KB")

    started = time.monotonic()
    try:
        markdown = await parse_pdf_with_mineru(path)
    except Exception as e:
        print(f"解析失败: {e}")
        return
    elapsed = time.monotonic() - started

    headers = _HEADER_RE.findall(markdown)
    h1 = sum(1 for lvl, _ in headers if lvl == "#")
    h2 = sum(1 for lvl, _ in headers if lvl == "##")
    h3 = sum(1 for lvl, _ in headers if lvl == "###")
    tables = markdown.count("<table")

    print(f"解析耗时: {elapsed:.1f}s")
    print(f"Markdown: {_fmt(len(markdown))} 字符")
    print(f"标题: h1={h1} h2={h2} h3={h3} | HTML 表格: {tables} 个")

    # 落盘一份，便于后续分析标题层级，避免重复调用 API 消耗额度
    dump_dir = os.path.join(os.path.dirname(__file__), "..", "data", "_test_md")
    os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write(markdown)

    result = split_with_parents(markdown, name, kb_doc_id=1, kb_id=1)
    print(f"切分策略: {result.strategy}")
    print(f"父块: {len(result.parents)} 个 | 子块: {len(result.children)} 个")

    if result.parents:
        sizes = [len(p.content) for p in result.parents]
        print(
            f"父块长度: 平均 {sum(sizes) // len(sizes)}  最大 {max(sizes)}  最小 {min(sizes)}"
        )
        oversized = [s for s in sizes if s > 4200]
        if oversized:
            print(f"  超过 4000 的父块: {len(oversized)} 个（应为 0）")

        print("章节路径样例（前 5 个）:")
        for p in result.parents[:5]:
            path = " > ".join(p.section_path) if p.section_path else "(无路径)"
            print(f"  - {path[:60]}  [{len(p.content)} 字符, {p.child_count} 子块]")

    if result.children:
        child_sizes = [len(c.page_content) for c in result.children]
        print(
            f"子块长度: 平均 {sum(child_sizes) // len(child_sizes)}  最大 {max(child_sizes)}"
        )
        broken_tables = sum(
            1
            for c in result.children
            if c.page_content.count("<table") != c.page_content.count("</table>")
        )
        if tables:
            print(f"  表格标签不闭合的子块: {broken_tables} 个（应为 0）")


async def main(paths: list[str]) -> int:
    for p in paths:
        if not os.path.exists(p):
            print(f"跳过不存在的文件: {p}")
            continue
        await test_one(p)
    print(f"\n{'=' * 70}\n测试完成")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/test_batch_parse.py <pdf...>")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
