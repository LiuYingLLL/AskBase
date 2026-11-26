"""
向量存储配置 — ChromaDB 持久化，单例模式
"""

from typing import Optional

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.config import settings
from app.rag.embeddings import get_embeddings

# 全局向量存储实例
_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    """获取 ChromaDB 向量存储实例（懒加载单例）

    telemetry 必须显式关闭：chromadb 0.5.x 只声明 `posthog>=2.4` 而没有上界，
    实际会装到 posthog 7.x，其 `capture()` 已改成关键字参数，
    于是每一次集合操作都在后台抛 TypeError。关掉后不再调用 posthog。
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        _vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _vectorstore


def reset_vectorstore():
    """重置向量存储实例（重新索引时使用）"""
    global _vectorstore
    _vectorstore = None


def is_index_broken(error: BaseException) -> bool:
    """判断异常是不是 Chroma 的索引损坏特征

    特征来自 hnswlib：集合里明明有数据，knn 查询却拼不出二维结果。
    这个错误跟「数据少」无关（k 已经按 count 收敛过），是索引状态坏了。
    """
    text = str(error)
    return "contigious 2D array" in text or "ef or M is too small" in text


def probe_index_health() -> tuple[bool, str]:
    """用集合里已有的一条向量做一次 knn，判断索引是否可用。

    不调用 embedding 接口：直接取回一条已存的向量当查询向量，零成本。
    返回 (是否健康, 说明)。
    """
    store = get_vectorstore()
    collection = store._collection
    try:
        count = collection.count()
    except Exception as e:  # noqa: BLE001 — 连 count 都读不出来，基本可以判定索引坏了
        return False, f"连集合条数都读不出来: {e}"

    if count == 0:
        return True, "空集合"

    data = collection.get(limit=1, include=["embeddings"])
    embeddings = data.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return True, f"{count} 条，取不到向量，跳过探测"

    try:
        collection.query(query_embeddings=[list(embeddings[0])], n_results=1)
    except Exception as e:  # noqa: BLE001
        if is_index_broken(e):
            return False, f"{count} 条，索引损坏: {e}"
        return True, f"{count} 条，探测遇到其他异常（不判定为损坏）: {e}"
    return True, f"{count} 条，检索正常"


def recreate_collection() -> int:
    """删除并重建向量集合，返回重建前集合里的条数。

    仅用于索引损坏后的恢复。调用方必须先确认原始文档还在（数据库记录 +
    上传文件），重建后需要重新执行索引才能把向量补回来。
    """
    global _vectorstore

    store = get_vectorstore()
    client = getattr(store, "_client", None)
    if client is None:
        raise RuntimeError("当前 Chroma 客户端不支持集合重建")

    try:
        count_before = store._collection.count()
    except Exception:  # noqa: BLE001 — 索引坏到连 count 都读不出来时按 0 处理
        count_before = 0

    name = settings.CHROMA_COLLECTION_NAME
    try:
        client.delete_collection(name)
    except Exception as e:  # noqa: BLE001 — 集合可能已经不存在，继续重建即可
        print(f"[VectorStore] 删除集合 {name} 时忽略异常: {e}")

    _vectorstore = None
    get_vectorstore()
    print(f"[VectorStore] 已重建集合 {name}（重建前 {count_before} 条）")
    return count_before
