"""父子块切分 — 两级切分：先按标题（或长度）切父块，再在父块内切子块

设计要点：

- 只有一条路径：有标题按标题切，没标题或章节过大按长度切虚拟父块。
  不做 A/B/C 三套并列分支，避免通用知识库的判断成本。

- 标题层级修复：MinerU 常把多级标题拍平成同一级。实测保险条款里
  333 个标题全是 h2（"第一部分""第六条""（一）""1.""（1）"混在一起），
  直接切会让父块碎成一行一块。这里用中文编号规则重建层级。

- 父块只切到指定深度（默认第 2 层，"条"级），更深的子标题留在父块正文里。
  否则"一条下面每一款都变成独立父块"，父块会比子块还小，回查失去意义。

- 短文档退化：整篇只切出一个父块且不超过一个子块尺寸时，不生成父块，
  子块直接带全文，避免为几千字的说明文档白存一份父块。

- 表格保护：超长 HTML 表格先按行拆成多个完整表格（各自带 <table></table>），
  再让子块分隔符优先在 </tr> 处断，避免标签被从中间切断。
"""

import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import settings


# 子块分隔符：优先表格行边界，再走中文语义边界
_CHILD_SEPARATORS = [
    "</tr>",
    "\n\n",
    "\n",
    "。",
    "；",
    ";",
    "，",
    ",",
    " ",
    "",
]

# 可由 PARENT_CHUNK_HEADER_DEPTH 截断使用
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_HEADER_PATTERN = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_HEADER_LINE_PATTERN = re.compile(r"^(#{1,6})[ \t]+(\S.*)$", re.MULTILINE)

# 中文法规/合同类文档的编号层级。顺序敏感：先匹配更具体的模式。
_NUMBERING_LEVELS = [
    (1, re.compile(r"^第[一二三四五六七八九十百千]+部分")),
    (1, re.compile(r"^第[一二三四五六七八九十百千]+章")),
    (2, re.compile(r"^第[一二三四五六七八九十百千]+节")),
    (2, re.compile(r"^第[一二三四五六七八九十百千]+条")),
    (2, re.compile(r"^\d+\.\d+")),          # 3.1 这种二级编号
    (3, re.compile(r"^[一二三四五六七八九十]+、")),
    (3, re.compile(r"^[（(][一二三四五六七八九十]+[）)]")),
    (4, re.compile(r"^\d+[\.、](?!\d)")),    # 1.住院医疗费用
    (5, re.compile(r"^[（(]\d+[）)]")),
]

_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL)
_TABLE_OPEN_PATTERN = re.compile(r"^<table[^>]*>", re.DOTALL)
_TR_ROW_PATTERN = re.compile(r"<tr[^>]*>.*?</tr>", re.DOTALL)

# 编号识别率低于这个比例，就认为不是编号型文档，不做层级改写
_REPAIR_RECOGNIZE_RATIO = 0.6
_REPAIR_MIN_HEADERS = 3


class DocumentTooLargeError(ValueError):
    """无结构的大文档不适合向量检索，直接拒绝入库"""


@dataclass
class ParentChunkDraft:
    """切分产出的父块草稿（未落库）"""

    id: str
    section_path: List[str]
    content: str
    child_count: int = 0


@dataclass
class SplitResult:
    """一次切分的完整产物"""

    parents: List[ParentChunkDraft] = field(default_factory=list)
    children: List[Document] = field(default_factory=list)
    # "header" 按标题切 / "virtual" 按长度切 / "flat" 退化为单块无父块
    strategy: str = "flat"

    @property
    def has_parents(self) -> bool:
        return bool(self.parents)


def has_markdown_headers(text: str) -> bool:
    """文档是否带 Markdown 标题结构"""
    return bool(_HEADER_PATTERN.search(text or ""))


def infer_numbering_level(title: str) -> Optional[int]:
    """按中文编号规则推断标题层级，认不出来返回 None

    第X部分/第X章 → 1，第X节/第X条 → 2，一、/(一) → 3，1. → 4，(1) → 5
    """
    cleaned = (title or "").strip()
    for level, pattern in _NUMBERING_LEVELS:
        if pattern.match(cleaned):
            return level
    return None


def _headers_are_flattened(markdown: str) -> bool:
    """判断标题层级是否被拍平：绝大多数标题挤在同一层，且更浅层几乎为空

    要求主导层级至少有 5 个标题且占比过半，避免把只有两三个标题的
    正常文档误判成拍平。
    """
    matches = _HEADER_LINE_PATTERN.findall(markdown)
    if len(matches) < _REPAIR_MIN_HEADERS:
        return False

    levels = [len(hashes) for hashes, _ in matches]
    counts = Counter(levels)
    dominant_level, dominant_count = counts.most_common(1)[0]

    if dominant_count < 5:
        return False
    if dominant_count / len(levels) < 0.6:
        return False

    shallower = sum(n for lvl, n in counts.items() if lvl < dominant_level)
    return shallower <= 3


def _levels_are_monotonic(levels: List[int]) -> bool:
    """层级序列是否大体不跳级（允许上升一级或下降任意级）

    正常文档偶尔会有一两处从"部分"直接跳到"一、"，严格禁止会误杀。
    这里按比例容忍：跳级次数不超过 5% 就算通过；短序列则要求零跳级，
    避免"h1 → h4"这种明显套错规则的文档被强行改写。
    """
    if not levels:
        return True

    jumps = 0
    prev = None
    for level in levels:
        if prev is not None and level > prev + 1:
            jumps += 1
        prev = level

    if len(levels) < 5:
        return jumps == 0
    return jumps / len(levels) <= 0.05


def repair_header_levels(markdown: str) -> str:
    """用中文编号规则重建标题层级

    只在"层级被拍平 + 编号识别率够高 + 重建后不跳级"时才动手，
    否则原样返回。识别不出编号的标题会被降级为普通正文（去掉 #），
    避免它们污染章节路径（典型情况：文首的公司名、条款名、注册号）。
    """
    if not _headers_are_flattened(markdown):
        return markdown

    matches = _HEADER_LINE_PATTERN.findall(markdown)
    inferred = [infer_numbering_level(text) for _, text in matches]
    recognized_levels = [lvl for lvl in inferred if lvl]
    if len(recognized_levels) < max(
        _REPAIR_MIN_HEADERS, len(matches) * _REPAIR_RECOGNIZE_RATIO
    ):
        return markdown

    if not _levels_are_monotonic(recognized_levels):
        return markdown

    def _replace(match: re.Match) -> str:
        title = match.group(2).strip()
        level = infer_numbering_level(title)
        if not level:
            # 认不出编号：降级为正文，不进章节路径
            return title
        return "#" * min(level, 4) + " " + title

    return _HEADER_LINE_PATTERN.sub(_replace, markdown)


def normalize_long_tables(text: str, max_chars: int) -> str:
    """把超长 HTML 表格按行拆成多个完整表格

    每个拆出来的片段都自带 <table> ... </table>，后续切分不会再切断标签。
    """
    if "<table" not in text:
        return text

    def _replace(match: re.Match) -> str:
        table = match.group(0)
        if len(table) <= max_chars:
            return table

        open_tag = _TABLE_OPEN_PATTERN.match(table)
        if not open_tag:
            return table
        head, tail = open_tag.group(0), "</table>"
        rows = _TR_ROW_PATTERN.findall(table)
        if not rows:
            return table

        budget = max(max_chars - len(head) - len(tail), 200)
        pieces: List[str] = []
        current: List[str] = []
        current_len = 0

        for row in rows:
            if current and current_len + len(row) > budget:
                pieces.append(head + "".join(current) + tail)
                current, current_len = [], 0
            current.append(row)
            current_len += len(row)
        if current:
            pieces.append(head + "".join(current) + tail)

        return "\n\n".join(pieces)

    return _TABLE_PATTERN.sub(_replace, text)


def _split_by_headers(
    text: str,
    depth: int = 2,
) -> List[Tuple[List[str], str]]:
    """按标题切到指定深度，返回 [(章节路径, 正文)]

    只切前 depth 层：更深的子标题留在正文里，避免父块被切得太碎。
    没有标题时返回 [([], 全文)]。
    """
    if not has_markdown_headers(text):
        return [([], text)]

    depth = max(1, min(depth, len(_HEADERS_TO_SPLIT_ON)))
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON[:depth],
        strip_headers=True,
    )
    sections = splitter.split_text(text)

    result: List[Tuple[List[str], str]] = []
    for section in sections:
        path = [v for v in section.metadata.values() if v]
        body = section.page_content.strip()
        if body:
            result.append((path, body))

    return result or [([], text)]


def _split_by_length(text: str, max_chars: int) -> List[str]:
    """按长度切块（父块超长时用）

    走与子块相同的表格感知切分：表格整块保留，不会被 </tr> 拆散。
    """
    return split_children(text, max_chars, 0)


def _split_at_table_boundary(text: str) -> List[Tuple[str, str]]:
    """把文本拆成 [("text", ...), ("table", ...), ...]，表格保持整体"""
    parts: List[Tuple[str, str]] = []
    last = 0
    for match in _TABLE_PATTERN.finditer(text):
        if match.start() > last:
            parts.append(("text", text[last:match.start()]))
        parts.append(("table", match.group(0)))
        last = match.end()
    if last < len(text):
        parts.append(("text", text[last:]))
    return parts


def _plain_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_CHILD_SEPARATORS,
        keep_separator="end",
    )
    return [p for p in splitter.split_text(text) if p.strip()]


def split_children(text: str, chunk_size: int, overlap: int) -> List[str]:
    """切子块，保证 HTML 表格不会被拆开

    纯文本仍按语义边界切；遇到表格时整块保留，允许单块略超 chunk_size。
    先前的做法只把 </tr> 放进分隔符，路径前缀加上表格一超长就会
    在表格中间断开，产生残缺的 <table> 标签。
    """
    if "<table" not in text:
        return _plain_split(text, chunk_size, overlap)

    chunks: List[str] = []
    buf = ""

    for kind, content in _split_at_table_boundary(text):
        if kind == "table":
            if buf and len(buf) + len(content) <= chunk_size:
                buf += content
                continue
            if buf.strip():
                chunks.append(buf)
            chunks.append(content)  # 表格独立成块，绝不拆
            buf = ""
            continue

        for piece in _plain_split(content, chunk_size, overlap):
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf.strip():
                    chunks.append(buf)
                buf = piece

    if buf.strip():
        chunks.append(buf)

    return [c for c in chunks if c.strip()]


def _render_parent_content(section_path: List[str], body: str) -> str:
    """父块正文前面补上章节路径，让模型知道这段属于哪一章"""
    if not section_path:
        return body
    return " > ".join(section_path) + "\n\n" + body


def split_with_parents(
    text: str,
    filename: str,
    kb_doc_id: int,
    kb_id: int | None = None,
    base_metadata: dict | None = None,
) -> SplitResult:
    """把一份文档切成父块 + 子块

    纯函数，不碰数据库。返回草稿，由调用方负责落库与写向量。
    """
    text = text or ""
    if not text.strip():
        return SplitResult()

    # 0. 无结构的大文档保护：这类文档切出来是上万个碎片，既检索不到也烧额度
    if not has_markdown_headers(text):
        limit = getattr(settings, "MAX_FLAT_DOC_CHARS", 500_000)
        if len(text) > limit:
            raise DocumentTooLargeError(
                f"文档约 {len(text):,} 字符且没有可识别的章节标题，"
                f"超过上限 {limit:,} 字符。这类文档（如整本费率表）不适合向量检索，"
                f"建议按章节拆分后上传，或改用数据库型知识库做结构化查询。"
            )

    max_parent_chars = max(settings.PARENT_CHUNK_MAX_CHARS, settings.CHUNK_SIZE)
    header_depth = getattr(settings, "PARENT_CHUNK_HEADER_DEPTH", 2)
    base_metadata = dict(base_metadata or {})

    # 0. 超长表格先拆成完整表格，避免后续切分切断标签
    text = normalize_long_tables(text, settings.CHUNK_SIZE)

    # 1. 标题层级已被拍平时先修复，否则按标题切会碎成一行一块
    repaired = repair_header_levels(text)
    has_headers = has_markdown_headers(repaired)
    strategy = "header" if has_headers else "virtual"

    # 2. 父块候选：标题章节，或标题章节里被长度再切的片
    raw_sections = _split_by_headers(repaired, depth=header_depth)
    candidates: List[Tuple[List[str], str]] = []
    for path, body in raw_sections:
        if len(body) <= max_parent_chars:
            candidates.append((path, body))
        else:
            for piece in _split_by_length(body, max_parent_chars):
                candidates.append((path, piece))

    # 3. 短文档退化：只有一个候选且不超过一个子块尺寸，就不做父子块
    if len(candidates) == 1 and len(candidates[0][1]) <= settings.CHUNK_SIZE:
        content = candidates[0][1]
        meta = dict(base_metadata)
        meta.update(
            {
                "filename": filename,
                "kb_doc_id": kb_doc_id,
                "chunk_index": 0,
                "total_chunks": 1,
                "kind": "doc",
                "chunk_strategy": "flat",
            }
        )
        if kb_id is not None:
            meta["kb_id"] = kb_id
        return SplitResult(
            parents=[],
            children=[Document(page_content=content, metadata=meta)],
            strategy="flat",
        )

    # 4. 每个父块内部切子块（表格整块保留，不被拆开）
    parents: List[ParentChunkDraft] = []
    children: List[Document] = []
    child_index = 0

    for path, body in candidates:
        parent_id = str(uuid.uuid4())
        parent_content = _render_parent_content(path, body)

        pieces = split_children(body, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        # 章节路径拼到子块正文前面，补足"这段属于什么责任"的语义
        path_prefix = (" > ".join(path) + "\n") if path else ""
        for piece in pieces:
            meta = dict(base_metadata)
            meta.update(
                {
                    "filename": filename,
                    "kb_doc_id": kb_doc_id,
                    "chunk_index": child_index,
                    "parent_id": parent_id,
                    "section_path": " > ".join(path),
                    "chunk_strategy": strategy,
                    "kind": "doc",
                }
            )
            if kb_id is not None:
                meta["kb_id"] = kb_id
            children.append(
                Document(page_content=path_prefix + piece, metadata=meta)
            )
            child_index += 1

        parents.append(
            ParentChunkDraft(
                id=parent_id,
                section_path=path,
                content=parent_content,
                child_count=len(pieces),
            )
        )

    for child in children:
        child.metadata["total_chunks"] = len(children)

    return SplitResult(parents=parents, children=children, strategy=strategy)
