"""
文档加载器 — 多格式文件分发加载

PDF 在配置了 MinerU 时优先走在线解析（保留表格/公式/扫描件语义），
失败自动降级到本地 PyPDFLoader，保证入库流程不中断。
其余格式仍走 LangChain 原生 loader。
"""

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from typing import List
import asyncio
import os

from app.rag.mineru_loader import is_mineru_configured, parse_pdf_with_mineru


LOADER_MAP = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".csv": "csv",
    ".xlsx": "excel",
    ".docx": "word",
}


def detect_file_type(file_path: str) -> str:
    """根据扩展名检测文件类型"""
    ext = os.path.splitext(file_path)[1].lower()
    return LOADER_MAP.get(ext, "unknown")


def should_use_mineru(file_path: str) -> bool:
    """是否该走 MinerU：已配置 + 是 PDF"""
    return is_mineru_configured() and detect_file_type(file_path) == "pdf"


def load_document(file_path: str) -> List[Document]:
    """根据文件类型加载文档（本地同步解析）"""
    file_type = detect_file_type(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if file_type == "text":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    elif file_type == "excel":
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    elif file_type == "word":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    return loader.load()


async def load_document_async(file_path: str) -> List[Document]:
    """异步加载文档 —— 入库链路的统一入口

    PDF 且 MinerU 可用时走在线解析；解析失败、超时、额度耗尽都降级到本地
    PyPDFLoader，降级对调用方透明。非 PDF 与未配置场景直接走本地解析，
    不发起任何网络请求。
    """
    if should_use_mineru(file_path):
        try:
            markdown = await parse_pdf_with_mineru(file_path)
            # 空内容等于没解析出东西，按失败处理，避免入库空文档
            if not markdown.strip():
                raise ValueError("MinerU 返回空内容")
            print(f"[Loader] MinerU 解析成功: {os.path.basename(file_path)}")
            return [
                Document(
                    page_content=markdown,
                    metadata={"source": file_path, "parser": "mineru"},
                )
            ]
        except Exception as e:  # noqa: BLE001 — 解析失败必须降级，不能中断入库
            print(f"[Loader] MinerU 解析失败，降级本地解析: {e}")

    # 本地 loader 是阻塞 IO，放到线程里跑，避免卡住事件循环
    return await asyncio.to_thread(load_document, file_path)
