"""配置模块测试 —— 只测「解析逻辑」和「必须成立的约束」

这里刻意不逐条断言配置默认值的字面量。
把 CHUNK_SIZE == 800 抄成断言，改默认值就得改测试，一个行为都验证不到；
真正会让线上出问题的是约束被破坏（比如重叠大于块大小）。
"""

import pytest

from app.config import Settings


class Test解析逻辑:
    """逗号分隔字符串 → 列表，这是真逻辑，值得测"""

    def test_允许的文件扩展名_解析正确(self):
        """场景：ALLOWED_EXTENSIONS 正确解析为列表"""
        settings = Settings(_env_file=None)
        extensions = settings.allowed_extensions_list
        assert "txt" in extensions
        assert "pdf" in extensions
        assert len(extensions) == len(set(extensions)), "不应有重复项"

    def test_CORS_来源_解析正确(self):
        """场景：CORS_ORIGINS 正确解析为列表"""
        origins = Settings(_env_file=None).cors_origins_list
        assert origins, "至少要有一个来源，否则前端全部被拦"
        assert all(o.startswith("http") for o in origins)

    def test_扩展名带空格和大小写_仍能解析(self):
        """场景：配置里写了 " PDF , Docx " → 归一化后可用"""
        settings = Settings(_env_file=None, ALLOWED_EXTENSIONS=" PDF , Docx ")
        parsed = settings.allowed_extensions_list
        assert "pdf" in parsed and "docx" in parsed
        assert all(e == e.strip() for e in parsed)


class Test关键约束:
    """这些约束一旦被破坏，功能会静默出错，比默认值漂移危险得多"""

    def test_切分重叠小于块大小(self):
        """场景：overlap >= chunk_size 时递归切分会死循环"""
        settings = Settings(_env_file=None)
        assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE

    def test_重排保留数量不超过候选数量(self):
        """场景：RERANK_OUTPUT_K > RERANK_TOP_N 时重排后取不满预期条数"""
        settings = Settings(_env_file=None)
        assert settings.RERANK_OUTPUT_K <= settings.RERANK_TOP_N

    def test_子块尺寸不超过父块上限(self):
        """场景：子块比父块还大会让父子块关系失去意义"""
        settings = Settings(_env_file=None)
        assert settings.CHUNK_SIZE <= settings.PARENT_CHUNK_MAX_CHARS

    def test_检索与超时参数为正(self):
        """场景：0 或负值会让检索、上传、解析直接失效"""
        settings = Settings(_env_file=None)
        assert settings.RETRIEVAL_TOP_K > 0
        assert settings.BM25_TOP_K > 0
        assert settings.MAX_UPLOAD_SIZE_MB > 0
        assert settings.JWT_EXPIRE_MINUTES > 0
        assert settings.MINERU_TIMEOUT > 0

    def test_相似度阈值在合法区间(self):
        """场景：阈值越界会让检索永远无结果或永远全过"""
        settings = Settings(_env_file=None)
        assert 0 <= settings.RETRIEVAL_SCORE_THRESHOLD <= 1
