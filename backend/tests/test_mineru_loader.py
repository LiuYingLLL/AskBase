"""测试 PDF 解析分流 — MinerU 优先、失败降级、非 PDF 不受影响"""

import asyncio

import pytest
from langchain_core.documents import Document

from app.config import settings
from app.rag import loader


def _run(coro):
    """在同步测试里跑协程，不依赖 pytest-asyncio 的标记配置"""
    return asyncio.run(coro)


@pytest.fixture
def pdf_file(tmp_path):
    """占位 .pdf 路径（解析函数已被 mock，不会真解析）"""
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4 placeholder")
    return str(p)


@pytest.fixture
def md_file(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# 标题\n\n正文内容", encoding="utf-8")
    return str(p)


@pytest.fixture
def mineru_on(monkeypatch):
    """开启 MinerU 并给一个假 token"""
    monkeypatch.setattr(settings, "MINERU_ENABLED", True)
    monkeypatch.setattr(settings, "MINERU_API_TOKEN", "fake-token")


class TestMinerU分流:

    def test_未配置token_不调用MinerU(self, monkeypatch, pdf_file):
        """token 为空时直接走本地解析，不发网络请求"""
        monkeypatch.setattr(settings, "MINERU_ENABLED", True)
        monkeypatch.setattr(settings, "MINERU_API_TOKEN", "")

        called = {"mineru": False}

        async def _fake_mineru(_):
            called["mineru"] = True
            return "# 不该被调用"

        monkeypatch.setattr(loader, "parse_pdf_with_mineru", _fake_mineru)
        monkeypatch.setattr(
            loader, "load_document", lambda _: [Document(page_content="本地解析")]
        )

        docs = _run(loader.load_document_async(pdf_file))

        assert called["mineru"] is False
        assert docs[0].page_content == "本地解析"

    def test_开关关闭_不调用MinerU(self, monkeypatch, pdf_file):
        """MINERU_ENABLED=false 时即使有 token 也不走 MinerU"""
        monkeypatch.setattr(settings, "MINERU_ENABLED", False)
        monkeypatch.setattr(settings, "MINERU_API_TOKEN", "fake-token")

        called = {"mineru": False}

        async def _fake_mineru(_):
            called["mineru"] = True
            return "# 不该被调用"

        monkeypatch.setattr(loader, "parse_pdf_with_mineru", _fake_mineru)
        monkeypatch.setattr(
            loader, "load_document", lambda _: [Document(page_content="本地解析")]
        )

        docs = _run(loader.load_document_async(pdf_file))

        assert called["mineru"] is False
        assert docs[0].page_content == "本地解析"

    def test_非PDF不走MinerU(self, monkeypatch, md_file):
        """md 文件即使配置了 MinerU 也不应发网络请求"""
        monkeypatch.setattr(settings, "MINERU_ENABLED", True)
        monkeypatch.setattr(settings, "MINERU_API_TOKEN", "fake-token")

        called = {"mineru": False}

        async def _fake_mineru(_):
            called["mineru"] = True
            return "# 不该被调用"

        monkeypatch.setattr(loader, "parse_pdf_with_mineru", _fake_mineru)

        docs = _run(loader.load_document_async(md_file))

        assert called["mineru"] is False
        assert "# 标题" in docs[0].page_content

    def test_配置齐全_PDF走MinerU(self, monkeypatch, mineru_on, pdf_file):
        """开关开 + token 有 + 是 PDF → 返回 MinerU 的 Markdown"""
        async def _fake_mineru(_):
            return "# 保险条款\n\n| 项目 | 金额 |\n| --- | --- |\n| 免赔额 | 1万 |"

        monkeypatch.setattr(loader, "parse_pdf_with_mineru", _fake_mineru)

        docs = _run(loader.load_document_async(pdf_file))

        assert len(docs) == 1
        assert docs[0].metadata["parser"] == "mineru"
        assert "免赔额" in docs[0].page_content

    def test_MinerU异常_降级本地解析(self, monkeypatch, mineru_on, pdf_file):
        """MinerU 抛异常时不中断，降级到本地 loader"""
        async def _boom(_):
            raise RuntimeError("网络超时")

        monkeypatch.setattr(loader, "parse_pdf_with_mineru", _boom)
        monkeypatch.setattr(
            loader, "load_document", lambda _: [Document(page_content="降级内容")]
        )

        docs = _run(loader.load_document_async(pdf_file))

        assert docs[0].page_content == "降级内容"
        assert docs[0].metadata.get("parser") != "mineru"

    def test_MinerU返回空白_视为失败降级(self, monkeypatch, mineru_on, pdf_file):
        """解析结果为空字符串时不能入库空文档，应降级"""
        async def _empty(_):
            return "   \n\n  "

        monkeypatch.setattr(loader, "parse_pdf_with_mineru", _empty)
        monkeypatch.setattr(
            loader, "load_document", lambda _: [Document(page_content="降级内容")]
        )

        docs = _run(loader.load_document_async(pdf_file))

        assert docs[0].page_content == "降级内容"

    def test_should_use_mineru_只认PDF(self, monkeypatch, mineru_on, pdf_file, md_file):
        """判定函数：只有 PDF 返回 True"""
        assert loader.should_use_mineru(pdf_file) is True
        assert loader.should_use_mineru(md_file) is False
