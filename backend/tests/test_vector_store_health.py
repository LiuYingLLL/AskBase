"""向量索引健康探测与损坏识别

背景：集合里明明有数据、count 也正常，但 hnsw 的 knn 查询会抛
"Cannot return the results in a contigious 2D array"，此时检索**不是**没有结果，
而是索引坏了。这类故障以前被降级成空列表，用户只看到「知识库中未找到资料」。
"""

import pytest

from app.rag import retriever as retriever_mod
from app.rag import vector_store as vs_mod
from app.rag.retriever import IndexBrokenError, _vector_search_sync
from app.rag.vector_store import is_index_broken, probe_index_health

BROKEN_MSG = (
    "Cannot return the results in a contigious 2D array. Probably ef or M is too small"
)


class _FakeCollection:
    def __init__(self, count=0, embeddings=None, query_error=None):
        self._count = count
        self._embeddings = embeddings if embeddings is not None else []
        self._query_error = query_error
        self.query_calls = 0

    def count(self):
        return self._count

    def get(self, limit=1, include=None):
        return {"embeddings": self._embeddings if self._count else []}

    def query(self, **kwargs):
        self.query_calls += 1
        if self._query_error:
            raise self._query_error
        return {"ids": [["x"]]}


class _FakeStore:
    def __init__(self, collection, search_error=None):
        self._collection = collection
        self._search_error = search_error
        self.search_calls = []

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None):
        self.search_calls.append(k)
        if self._search_error:
            raise self._search_error
        return []


class Test损坏识别:

    def test_典型索引错误_识别为损坏(self):
        assert is_index_broken(RuntimeError(BROKEN_MSG)) is True
        assert is_index_broken(RuntimeError("ef or M is too small")) is True

    def test_其他异常_不误判(self):
        assert is_index_broken(RuntimeError("connection reset")) is False
        assert is_index_broken(ValueError("bad filter")) is False


class Test健康探测:

    def test_空集合_视为健康(self, monkeypatch):
        store = _FakeStore(_FakeCollection(count=0))
        monkeypatch.setattr(vs_mod, "_vectorstore", store)

        healthy, detail = probe_index_health()

        assert healthy is True
        assert "空集合" in detail
        # 空集合不应触发查询，避免拿不到向量却硬查
        assert store._collection.query_calls == 0

    def test_有数据且可查询_视为健康(self, monkeypatch):
        col = _FakeCollection(count=3, embeddings=[[0.1, 0.2]])
        store = _FakeStore(col)
        monkeypatch.setattr(vs_mod, "_vectorstore", store)

        healthy, detail = probe_index_health()

        assert healthy is True
        assert col.query_calls == 1
        assert "3 条" in detail

    def test_有数据但查询报索引错_判定损坏(self, monkeypatch):
        col = _FakeCollection(count=5, embeddings=[[0.1, 0.2]], query_error=RuntimeError(BROKEN_MSG))
        store = _FakeStore(col)
        monkeypatch.setattr(vs_mod, "_vectorstore", store)

        healthy, detail = probe_index_health()

        assert healthy is False
        assert "索引损坏" in detail

    def test_查询报其他异常_不判定损坏(self, monkeypatch):
        """探测遇到非索引类异常时宁可放过，避免无谓清空集合"""
        col = _FakeCollection(count=5, embeddings=[[0.1, 0.2]], query_error=RuntimeError("timeout"))
        store = _FakeStore(col)
        monkeypatch.setattr(vs_mod, "_vectorstore", store)

        healthy, _ = probe_index_health()

        assert healthy is True


class Test检索边界:

    def test_空集合_直接返回空且不查询(self, monkeypatch):
        store = _FakeStore(_FakeCollection(count=0))
        monkeypatch.setattr(retriever_mod, "get_vectorstore", lambda: store)

        assert _vector_search_sync("任意问题", 5) == []
        assert store.search_calls == []

    def test_k_按集合条数收敛(self, monkeypatch):
        """集合只有 2 条时不能按 5 条去查，否则 Chroma 报 contigious 错误"""
        store = _FakeStore(_FakeCollection(count=2))
        monkeypatch.setattr(retriever_mod, "get_vectorstore", lambda: store)

        _vector_search_sync("任意问题", 5)

        assert store.search_calls == [2]

    def test_收敛后仍报索引错_抛专用异常(self, monkeypatch):
        """count=1、k 也收敛到 1 还报错，只能是索引坏了，要能被上层区分"""
        store = _FakeStore(_FakeCollection(count=1), search_error=RuntimeError(BROKEN_MSG))
        monkeypatch.setattr(retriever_mod, "get_vectorstore", lambda: store)

        with pytest.raises(IndexBrokenError) as exc:
            _vector_search_sync("任意问题", 5)

        assert exc.value.count == 1
        assert "重建索引" in str(exc.value)

    def test_非索引类异常_原样抛出(self, monkeypatch):
        store = _FakeStore(_FakeCollection(count=3), search_error=RuntimeError("network down"))
        monkeypatch.setattr(retriever_mod, "get_vectorstore", lambda: store)

        with pytest.raises(RuntimeError, match="network down"):
            _vector_search_sync("任意问题", 5)
