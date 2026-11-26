"""测试父子块切分与父块回查

覆盖两类行为：
1. 切分 —— 有标题 / 无标题 / 超长章节 / 短文档退化 / 表格保护
2. 回查 —— 父块替换、同父块去重、无父块原样保留、父块缺失降级
"""

import pytest
from langchain_core.documents import Document

from app.config import settings
from app.rag import retriever
from app.rag.parent_child import (
    has_markdown_headers,
    split_with_parents,
)
from app.rag.parent_chunk_store import loads_path


MARKDOWN_WITH_HEADERS = """# 第三章 保险责任

本章说明各项保险金的给付条件。

## 3.1 身故保险金

被保险人在保险期间内身故，本公司按基本保额给付身故保险金。

## 3.2 重大疾病保险金

被保险人初次确诊约定重大疾病，本公司按基本保额给付重大疾病保险金。
"""


class Test切分结构:

    def test_有标题_按章节切父块(self):
        """有标题的文档 → 每个章节一个父块，路径正确"""
        result = split_with_parents(
            MARKDOWN_WITH_HEADERS, "条款.md", kb_doc_id=1, kb_id=2
        )

        assert result.strategy == "header"
        assert len(result.parents) >= 2
        # 至少有一级 h2 父块，路径里能看出所属章节
        paths = [" > ".join(p.section_path) for p in result.parents]
        assert any("3.1" in p for p in paths)
        assert any("3.2" in p for p in paths)

    def test_子块都带parent_id和章节路径(self):
        """每个子块都要能追溯到父块，且正文里带章节信息"""
        result = split_with_parents(
            MARKDOWN_WITH_HEADERS, "条款.md", kb_doc_id=1, kb_id=2
        )

        parent_ids = {p.id for p in result.parents}
        assert result.children
        for child in result.children:
            pid = child.metadata.get("parent_id")
            assert pid in parent_ids
            assert child.metadata.get("chunk_strategy") == "header"
            assert child.metadata.get("kb_doc_id") == 1

    def test_父块的子块数量对得上(self):
        """父块声明的 child_count 要等于实际挂在它下面的子块数"""
        result = split_with_parents(
            MARKDOWN_WITH_HEADERS, "条款.md", kb_doc_id=1, kb_id=2
        )

        counted = {}
        for child in result.children:
            pid = child.metadata["parent_id"]
            counted[pid] = counted.get(pid, 0) + 1

        for parent in result.parents:
            assert parent.child_count == counted.get(parent.id, 0)

    def test_无标题_切虚拟父块(self):
        """无标题文档（如解析出的纯表格）→ 走长度切分，不报错"""
        plain = "这是没有标题的正文。" * 800  # 远超父块上限

        result = split_with_parents(plain, "表格.md", kb_doc_id=3, kb_id=4)

        assert result.strategy == "virtual"
        assert len(result.parents) >= 2
        for parent in result.parents:
            assert len(parent.content) <= settings.PARENT_CHUNK_MAX_CHARS + 200

    def test_超长章节_切成多个虚拟父块(self):
        """单个章节超过父块上限 → 拆成多个父块，共享同一路径"""
        long_section = "# 长章节\n\n" + ("这是很长的一段内容。" * 800)

        result = split_with_parents(long_section, "长文.md", kb_doc_id=5, kb_id=6)

        assert len(result.parents) >= 2
        # 拆出来的父块应共享同一章节路径
        paths = {tuple(p.section_path) for p in result.parents}
        assert len(paths) == 1

    def test_短文档_退化为单块无父块(self):
        """整篇不超过一个子块 → 不生成父块，省一份存储"""
        short = "很短的一段说明。"

        result = split_with_parents(short, "短.md", kb_doc_id=7, kb_id=8)

        assert result.strategy == "flat"
        assert result.parents == []
        assert len(result.children) == 1
        assert result.children[0].metadata["chunk_strategy"] == "flat"
        assert "parent_id" not in result.children[0].metadata

    def test_空文本_返回空结果(self):
        result = split_with_parents("   \n  ", "空.md", kb_doc_id=9, kb_id=10)
        assert result.children == []
        assert result.parents == []

    def test_HTML表格_不切断标签(self):
        """超长 HTML 表格切分时，子块不应以半个标签结尾"""
        rows = "".join(
            f"<tr><td>第{i}行</td><td>数值{i}</td></tr>" for i in range(400)
        )
        table = f"<table>{rows}</table>"

        result = split_with_parents(table, "费率表.md", kb_doc_id=11, kb_id=12)

        assert result.children
        for child in result.children:
            text = child.page_content.strip()
            # 以 </tr> 结尾说明在行边界断开，没有把一行切两半
            assert text.endswith("</tr>") or len(text) < settings.CHUNK_SIZE

    def test_has_markdown_headers_判定(self):
        assert has_markdown_headers("# 标题\n正文") is True
        assert has_markdown_headers("## 二级标题\n正文") is True
        assert has_markdown_headers("普通文本\n没有标题") is False
        assert has_markdown_headers("#话题标签\n这不是标题") is False


class Test父块回查:

    @pytest.fixture
    def fake_parents(self, monkeypatch):
        """用内存字典替掉数据库查询"""
        rows = {
            "p1": type("Row", (), {
                "id": "p1",
                "filename": "条款.md",
                "section_path": '["第三章", "3.1 身故保险金"]',
                "content": "第三章 > 3.1 身故保险金\n\n完整章节内容，含例外条件。",
                "kb_id": 2,
                "kb_doc_id": 1,
            })(),
        }

        async def _fake(ids):
            return {k: v for k, v in rows.items() if k in set(ids)}

        monkeypatch.setattr(retriever, "load_parents_isolated", _fake)
        return rows

    def test_命中子块_替换成父块(self, monkeypatch, fake_parents):
        """回查后内容变成父块全文，章节路径进元数据"""
        monkeypatch.setattr(settings, "PARENT_CHUNK_ENABLED", True)
        hits = [
            (Document(page_content="半句话", metadata={"parent_id": "p1", "filename": "条款.md"}), 0.9)
        ]

        out = run(retriever.expand_to_parents(hits))

        assert len(out) == 1
        doc, score = out[0]
        assert "完整章节内容" in doc.page_content
        assert doc.metadata["section_path"] == "第三章 > 3.1 身故保险金"
        assert score == 0.9

    def test_同一父块的多个子块_只留一条(self, monkeypatch, fake_parents):
        """多个子块指向同一父块 → 合并成一条，保留首次出现的位置"""
        monkeypatch.setattr(settings, "PARENT_CHUNK_ENABLED", True)
        hits = [
            (Document(page_content="块A", metadata={"parent_id": "p1"}), 0.9),
            (Document(page_content="块B", metadata={"parent_id": "p1"}), 0.8),
        ]

        out = run(retriever.expand_to_parents(hits))

        assert len(out) == 1
        assert out[0][1] == 0.9

    def test_无parent_id的块_原样保留(self, monkeypatch, fake_parents):
        """老数据 / 问答集 / 爬虫数据没有 parent_id，不能被吞掉"""
        monkeypatch.setattr(settings, "PARENT_CHUNK_ENABLED", True)
        hits = [
            (Document(page_content="老数据", metadata={"filename": "旧.md"}), 0.7),
            (Document(page_content="子块", metadata={"parent_id": "p1"}), 0.9),
        ]

        out = run(retriever.expand_to_parents(hits))

        contents = [d.page_content for d, _ in out]
        assert any("老数据" in c for c in contents)
        assert any("完整章节内容" in c for c in contents)

    def test_父块查不到_退回子块(self, monkeypatch, fake_parents):
        """父块被误删时不能报错，退回子块内容"""
        monkeypatch.setattr(settings, "PARENT_CHUNK_ENABLED", True)
        hits = [
            (Document(page_content="子块内容", metadata={"parent_id": "missing"}), 0.6)
        ]

        out = run(retriever.expand_to_parents(hits))

        assert len(out) == 1
        assert out[0][0].page_content == "子块内容"

    def test_开关关闭_不回查(self, monkeypatch, fake_parents):
        """PARENT_CHUNK_ENABLED=false 时直接返回原结果"""
        monkeypatch.setattr(settings, "PARENT_CHUNK_ENABLED", False)
        hits = [
            (Document(page_content="子块", metadata={"parent_id": "p1"}), 0.9)
        ]

        out = run(retriever.expand_to_parents(hits))

        assert out[0][0].page_content == "子块"


class Test路径解析:

    def test_正常JSON_解析成列表(self):
        assert loads_path('["第三章", "3.1"]') == ["第三章", "3.1"]

    def test_脏数据_返回空列表(self):
        assert loads_path("not json") == []
        assert loads_path(None) == []
        assert loads_path("") == []


class Test标题层级修复:
    """MinerU 常把多级标题拍平成同一级，需要用中文编号规则重建"""

    # 模拟真实解析结果：h1 是文档头（公司名+注册号），底下全是 h2
    FLATTENED = """# 某某保险股份有限公司个人医疗险条款注册号：C0001

## 第一部分 总则

## 第一条 合同构成

本保险合同由保险条款、投保单组成。

## 第二条 投保人

投保人应为具有完全民事行为能力的被保险人本人。

## 第二部分 保障内容

## 第三条 保险责任

在本合同保险期间内，本公司承担下列保险责任。

### （一）一般医疗保险金

## 1.住院医疗费用

被保险人经医院诊断必须住院治疗的。

## 2.特殊门诊医疗费用

被保险人经医院诊断必须接受特殊门诊治疗的。
"""

    def test_识别中文编号层级(self):
        from app.rag.parent_child import infer_numbering_level

        assert infer_numbering_level("第一部分 总则") == 1
        assert infer_numbering_level("第三条 保险责任") == 2
        assert infer_numbering_level("（一）一般医疗保险金") == 3
        assert infer_numbering_level("1.住院医疗费用") == 4
        assert infer_numbering_level("（1）给付条件") == 5
        assert infer_numbering_level("某某保险股份有限公司") is None

    def test_拍平的层级被重建(self):
        from app.rag.parent_child import repair_header_levels

        repaired = repair_header_levels(self.FLATTENED)

        # 文档头（无编号）被降级为正文，不再污染路径
        assert not repaired.startswith("# 某某保险")
        # 编号层级被还原
        assert "\n# 第一部分 总则" in repaired
        assert "\n## 第一条 合同构成" in repaired
        assert "\n### （一）一般医疗保险金" in repaired
        assert "\n#### 1.住院医疗费用" in repaired

    def test_正常多级文档不被改写(self):
        from app.rag.parent_child import repair_header_levels

        normal = "# 第三章\n\n## 3.1 身故保险金\n\n正文一\n\n## 3.2 重疾保险金\n\n正文二\n"
        assert repair_header_levels(normal) == normal

    def test_重建后父块只切到条级(self):
        """父块深度默认 2：一条一个父块，条下面的款留在父块正文里"""
        from app.rag.parent_child import repair_header_levels

        repaired = repair_header_levels(self.FLATTENED)
        result = split_with_parents(repaired, "条款.pdf", kb_doc_id=20, kb_id=21)

        paths = [" > ".join(p.section_path) for p in result.parents]
        # 路径里能看到"部分 > 条"两级
        assert any("第二部分 保障内容 > 第三条 保险责任" in p for p in paths)
        # 更深层的"（一）""1." 不应单独成为父块
        assert not any("（一）一般医疗保险金" in p for p in paths)
        assert not any("1.住院医疗费用" in p for p in paths)


class Test表格保护:
    """超长 HTML 表格不能被切断"""

    def test_超长表格拆成多个完整表格(self):
        from app.rag.parent_child import normalize_long_tables

        rows = "".join(f"<tr><td>第{i}行</td><td>值{i}</td></tr>" for i in range(200))
        table = f"<table>{rows}</table>"

        normalized = normalize_long_tables(table, 800)
        pieces = normalized.split("\n\n")

        assert len(pieces) > 1
        for piece in pieces:
            assert piece.startswith("<table>")
            assert piece.endswith("</table>")

    def test_短表格原样保留(self):
        from app.rag.parent_child import normalize_long_tables

        table = "<table><tr><td>a</td></tr></table>"
        assert normalize_long_tables(table, 800) == table

    def test_切分后表格标签全部闭合(self):
        rows = "".join(f"<tr><td>第{i}行</td><td>值{i}</td></tr>" for i in range(200))
        text = f"# 费率表\n\n<table>{rows}</table>\n\n## 说明\n\n以上为费率。"

        result = split_with_parents(text, "费率.md", kb_doc_id=22, kb_id=23)

        for child in result.children:
            opened = child.page_content.count("<table")
            closed = child.page_content.count("</table>")
            assert opened == closed, f"标签不闭合: {child.page_content[:80]}"


class Test大文档保护:
    """无结构的大文档会切出上万块，必须拒绝"""

    def test_无标题超大文档被拒绝(self, monkeypatch):
        from app.rag.parent_child import DocumentTooLargeError

        monkeypatch.setattr(settings, "MAX_FLAT_DOC_CHARS", 1000)
        huge_plain = "没有标题的正文内容。" * 200  # 2000 字符

        with pytest.raises(DocumentTooLargeError) as exc:
            split_with_parents(huge_plain, "大.csv", kb_doc_id=30, kb_id=31)

        assert "不适合向量检索" in str(exc.value)

    def test_有标题的大文档不拒绝(self, monkeypatch):
        """有章节结构的大文档按标题切，不该被拦"""
        monkeypatch.setattr(settings, "MAX_FLAT_DOC_CHARS", 1000)
        sections = "".join(
            f"# 第{i}章\n\n{'正文内容。' * 100}\n\n" for i in range(5)
        )

        result = split_with_parents(sections, "书.md", kb_doc_id=32, kb_id=33)

        assert result.strategy == "header"
        assert len(result.parents) >= 5

    def test_小文档不受影响(self, monkeypatch):
        monkeypatch.setattr(settings, "MAX_FLAT_DOC_CHARS", 1000)
        small = "很短的一段说明。"

        result = split_with_parents(small, "小.md", kb_doc_id=34, kb_id=35)

        assert result.strategy == "flat"
        assert len(result.children) == 1


class Test父块仓储:
    """用临时 SQLite 验证建表、覆盖写入、回查、删除"""

    def test_写入回查覆盖与删除(self, tmp_path):
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.database import Base
        # 注册全部 ORM 模型，否则 agents.model_id 等外键找不到目标表（与 app/main.py 同一集合）
        import app.models  # noqa: F401
        from app.models.llm_model import LLMModel  # noqa: F401
        from app.models.skill import Skill  # noqa: F401
        from app.models.mcp_server import MCPServer  # noqa: F401
        from app.models.agent import AgentTool  # noqa: F401
        from app.rag.parent_child import ParentChunkDraft
        from app.rag.parent_chunk_store import ParentChunkStore

        async def _flow():
            engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path/'parent.db'}")
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, expire_on_commit=False)

            drafts = [
                ParentChunkDraft(
                    id="11111111-1111-1111-1111-111111111111",
                    section_path=["第三章", "3.1"],
                    content="第三章 > 3.1\n\n正文一",
                    child_count=2,
                ),
                ParentChunkDraft(
                    id="22222222-2222-2222-2222-222222222222",
                    section_path=["第三章", "3.2"],
                    content="第三章 > 3.2\n\n正文二",
                    child_count=1,
                ),
            ]

            async with factory() as db:
                store = ParentChunkStore(db)
                written = await store.replace_for_document(
                    1, drafts, "条款.md", kb_id=2
                )
                await db.commit()
                assert written == 2

                got = await store.load_many([d.id for d in drafts])
                assert len(got) == 2
                assert loads_path(got[drafts[0].id].section_path) == ["第三章", "3.1"]

                # 覆盖写入应幂等，不产生重复行
                await store.replace_for_document(1, drafts, "条款.md", kb_id=2)
                await db.commit()
                again = await store.load_many([d.id for d in drafts])
                assert len(again) == 2

                # 删除后应查不到
                await store.delete_for_document(1)
                await db.commit()
                assert await store.load_many([d.id for d in drafts]) == {}

            await engine.dispose()

        run(_flow())


def run(coro):
    import asyncio

    return asyncio.run(coro)
