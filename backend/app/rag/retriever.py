"""
检索器 — 向量语义检索 + BM25 关键词检索（RRF 融合）+ 可选重排与缓存
"""

import asyncio
import functools
import logging
import traceback
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.rag.bm25_index import get_bm25_index
from app.rag.cache import retrieval_cache
from app.rag.fusion import attach_display_scores, rrr_fuse
from app.rag.parent_chunk_store import load_parents_isolated, loads_path
from app.rag.reranker import create_reranker
from app.rag.vector_store import get_vectorstore, is_index_broken

logger = logging.getLogger(__name__)

# ChromaDB 并发检索锁（限制同时检索数，避免文件I/O竞争）
_search_semaphore = asyncio.Semaphore(10)


class IndexBrokenError(RuntimeError):
    """向量索引损坏：集合里有数据（count>0）但 knn 查询拼不出结果。

    与「没检索到相关内容」是两回事，不能当成空结果返回，
    否则用户只会看到「知识库中未找到资料」而不知道是索引坏了。
    """

    def __init__(self, count: int):
        super().__init__(
            f"向量索引已损坏（集合内有 {count} 条数据但无法检索），"
            "请在知识库页面执行「重建索引」"
        )
        self.count = count


def _cache_key(
    query: str,
    top_k: int,
    score_threshold: float,
    kb_doc_ids: Optional[List[int]] = None,
    kb_ids: Optional[List[int]] = None,
) -> tuple:
    """构造缓存键（配置开关变化时缓存自然失效）"""
    scope_doc = tuple(sorted(int(x) for x in kb_doc_ids)) if kb_doc_ids else None
    scope_kb = tuple(sorted(int(x) for x in kb_ids)) if kb_ids else None
    return (
        query,
        scope_doc,
        scope_kb,
        top_k,
        score_threshold,
        settings.HYBRID_ENABLED,
        settings.RERANK_ENABLED,
        settings.RERANK_MODE,
        settings.PARENT_CHUNK_ENABLED,
    )


async def _search_async(query: str, top_k: int, filter_dict: Optional[dict] = None):
    """在独立线程中执行 ChromaDB 检索，避免阻塞事件循环"""
    async with _search_semaphore:
        return await asyncio.to_thread(_vector_search_sync, query, top_k, filter_dict)


def _vector_search_sync(
    query: str, top_k: int, filter_dict: Optional[dict] = None
) -> List[Tuple[Document, float]]:
    """Chroma 向量检索（同步）。

    集合里的元素少于请求的 k 时，Chroma 会抛
    "Cannot return the results in a contigious 2D array. Probably ef or M is too small"。
    所以先按集合实际条数收敛 k；带过滤条件时实际候选可能更少，再退到 k=1 重试。
    收敛后仍报同一个错，说明索引本身坏了（不是数据少），直接把异常抛给上层，
    由上层打日志并让用户走重建索引，绝不静默当成「没有资料」。
    """
    vectorstore = get_vectorstore()
    total = vectorstore._collection.count()
    if total == 0:
        return []

    kwargs = {"filter": filter_dict} if filter_dict else {}
    k = min(top_k, total)
    try:
        return vectorstore.similarity_search_with_relevance_scores(query, k=k, **kwargs)
    except Exception as e:  # noqa: BLE001 — 只对已知的 Chroma 边界错误做降级
        if not is_index_broken(e):
            raise
        if k <= 1:
            raise IndexBrokenError(total) from e
        # 过滤后实际命中可能不足 total，退到单条重试
        try:
            return vectorstore.similarity_search_with_relevance_scores(query, k=1, **kwargs)
        except Exception as e2:  # noqa: BLE001
            raise IndexBrokenError(total) from e2


def _bm25_search_sync(
    query: str,
    top_k: int,
    kb_doc_ids: Optional[List[int]] = None,
    kb_ids: Optional[List[int]] = None,
) -> List[Tuple[Document, float]]:
    """BM25 检索（同步执行，调用方负责 to_thread 包装）"""
    return get_bm25_index().search(query, top_k, kb_doc_ids, kb_ids)


def _build_scope_filter(
    kb_doc_ids: Optional[List[int]],
    kb_ids: Optional[List[int]],
) -> Optional[dict]:
    """构造 ChromaDB 过滤条件：kb_ids（知识库维度）优先，其次 kb_doc_ids（旧兼容）"""
    if kb_ids:
        return {"kb_id": {"$in": [int(x) for x in kb_ids]}}
    if kb_doc_ids:
        return {"kb_doc_id": {"$in": [int(x) for x in kb_doc_ids]}}
    return None


async def expand_to_parents(
    hits: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """把命中的子块替换成它的完整父块，同一父块只保留第一次出现

    - 没有 parent_id 的块（老数据 / 问答集 / 知识点 / 爬虫灌入）原样保留
    - 父块查不到时退回子块内容，不让检索链路报错
    - 多个子块指向同一父块时会合并，返回条数变少属预期行为
    """
    if not settings.PARENT_CHUNK_ENABLED or not hits:
        return hits

    ordered_ids: List[str] = []
    seen = set()
    for doc, _ in hits:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            ordered_ids.append(pid)

    if not ordered_ids:
        return hits

    parents = await load_parents_isolated(ordered_ids)

    expanded: List[Tuple[Document, float]] = []
    emitted = set()
    for doc, score in hits:
        pid = doc.metadata.get("parent_id")
        parent = parents.get(pid) if pid else None

        if parent is None:
            # 无父块或父块缺失：保留原子块，保证检索不中断
            expanded.append((doc, score))
            continue

        if pid in emitted:
            continue
        emitted.add(pid)

        metadata = dict(doc.metadata)
        metadata.update({
            "filename": parent.filename,
            "parent_id": pid,
            "section_path": " > ".join(loads_path(parent.section_path)),
            "kb_id": parent.kb_id,
            "kb_doc_id": parent.kb_doc_id,
            "kind": "doc",
        })
        expanded.append(
            (Document(page_content=parent.content, metadata=metadata), score)
        )

    return expanded


async def retrieve_with_scores(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    kb_doc_ids: Optional[List[int]] = None,
    kb_ids: Optional[List[int]] = None,
    use_cache: bool = True,
) -> List[Tuple[Document, float]]:
    """检索并返回 (Document, score) 元组，用于引用展示（异步）

    流程：缓存查询 → 向量路召回 → BM25 路召回（可选）→ RRF 融合 → 重排（可选）→ 展示分数

    作用域：
      - kb_ids 非空时，仅在指定知识库集合内检索（知识库维度，Phase 6 起主用）；
      - kb_doc_ids 非空时，按文档维度过滤（旧调用兼容）；
      - 均为空时等价于全库检索，与历史行为完全一致。
    """
    try:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold or settings.RETRIEVAL_SCORE_THRESHOLD

        # 1. 缓存查询
        use_cache = use_cache and settings.CACHE_ENABLED
        key = _cache_key(query, top_k, score_threshold, kb_doc_ids, kb_ids)
        if use_cache:
            cached = retrieval_cache.get(key)
            if cached is not None:
                return cached

        # 2. 向量路（重排开启时扩大召回量）
        recall_k = settings.RERANK_TOP_N if settings.RERANK_ENABLED else top_k
        filter_dict = _build_scope_filter(kb_doc_ids, kb_ids)
        vector_results = await _search_async(query, recall_k, filter_dict)
        # 阈值语义（P4 优化）：召回阶段不过滤——弱相关候选也进入融合池，
        # 由 RRF + 重排决定最终排序；score_threshold 仅由调用方用于"是否有依据"判定
        # （见 chain.py 的 no_results 逻辑），避免误杀潜在正确答案。

        # 3. BM25 路（可选；构建/检索异常时优雅降级为纯向量检索）
        bm25_results: List[Tuple[Document, float]] = []
        if settings.HYBRID_ENABLED:
            try:
                async with _search_semaphore:
                    bm25_results = await asyncio.to_thread(
                        functools.partial(
                            _bm25_search_sync, query, settings.BM25_TOP_K, kb_doc_ids, kb_ids
                        )
                    )
            except Exception as e:
                print(f"[Retriever] BM25 检索降级（纯向量）: {e}")
                bm25_results = []

        # 4. RRF 融合
        hits = rrr_fuse(vector_results, bm25_results, settings.HYBRID_RRF_K)

        # 5. 重排（可选）
        if settings.RERANK_ENABLED:
            reranker = create_reranker()
            candidates = hits[:settings.RERANK_TOP_N]
            hits = await reranker.rerank(query, candidates, settings.RERANK_OUTPUT_K)
        else:
            hits = hits[:top_k]

        # 6. 展示分数 → 回查父块 → 写缓存
        result = attach_display_scores(hits)
        result = await expand_to_parents(result)
        if use_cache:
            retrieval_cache.set(key, result)
        return result
    except IndexBrokenError as e:
        # 索引损坏不是「没检索到」，是系统故障。这里不向上抛（问答不能 500），
        # 但必须留下能定位的错误日志，并在重建索引时可被识别。
        logger.error("[Retriever] %s", e)
        return []
    except Exception as e:
        # 其余异常降级为空结果，但不能静默：日志要能区分「没召回」和「出错了」
        logger.error(
            "[Retriever] 检索异常，已降级为空结果: %s: %s\n%s",
            type(e).__name__,
            e,
            traceback.format_exc(),
        )
        return []


async def retrieve_similar_chunks(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    kb_doc_ids: Optional[List[int]] = None,
    kb_ids: Optional[List[int]] = None,
) -> List[Document]:
    """从向量存储中检索与查询最相似的文档片段（异步，混合检索的薄封装）"""
    results_with_scores = await retrieve_with_scores(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        kb_doc_ids=kb_doc_ids,
        kb_ids=kb_ids,
    )
    return [doc for doc, _ in results_with_scores]
