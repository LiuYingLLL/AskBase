"""
应用配置管理 — 使用 pydantic-settings 从 .env 文件和环境变量加载配置
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os


class Settings(BaseSettings):
    """应用全局配置"""

    # === 应用 ===
    APP_NAME: str = "AskBase"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # === LLM（OpenAI 兼容端点，如 DeepSeek / 百炼） ===
    LLM_API_KEY: str = ""
    LLM_API_BASE: str = "https://api.deepseek.com"
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.3
    # 流式请求是否带 usage 参数（百炼必须带才有用量；DeepSeek 带不带都返回）。
    # 遇到严格校验该参数的兼容端点报 400 时，可置 false 兜底。
    LLM_STREAM_USAGE: bool = True

    # === 联网搜索（Exa Answer，可选：填 key 后 web_search 优先走 Exa，失效自动回退百度/Bing） ===
    EXA_API_KEY: str = ""

    # === Embedding（OpenAI 兼容端点，如百炼 / 硅基流动） ===
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL: str = "text-embedding-v3"

    # === 数据库 ===
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/app.db"

    # === JWT ===
    JWT_SECRET: str = "change-this-to-a-random-string-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24小时

    # === ChromaDB ===
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    # 历史默认值曾是 ecommerce_kb（早期电商题材遗留）。注意：改这个默认值不会迁移已有集合，
    # 已有向量数据仍存在旧名下，换名会导致检索不到；要改名需自行迁移数据。
    CHROMA_COLLECTION_NAME: str = "askbase_kb"

    # === RAG 参数 ===
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_TOP_K: int = 5

    # === 父子块切分 ===
    # 开启后：按标题切父块 → 父块内切子块 → 检索命中子块后回查完整父块。
    # 关闭后退回"一刀"固定长度切分，用于快速回滚。
    PARENT_CHUNK_ENABLED: bool = True
    # 单个父块的内容上限（字符）。超过则按长度切成虚拟父块。
    # 太小等于没回查，太大容易把模型上下文塞满，4000 是折中值。
    PARENT_CHUNK_MAX_CHARS: int = 4000
    # 父块切到第几级标题（1-4）。默认 2，即"章/条"级：
    # 更深的子标题留在父块正文里，避免父块碎成一行一块。
    PARENT_CHUNK_HEADER_DEPTH: int = 2
    # 无标题文档的字符上限。超过直接拒绝入库：
    # 实测一份 238 万字符的费率表（15 个巨型表格、零标题）会切出上万块，
    # 既不可检索也严重消耗 embedding 额度。这类文档应走结构化查询而非向量检索。
    MAX_FLAT_DOC_CHARS: int = 500_000
    # 向量路相似度阈值：混合检索（BM25 兜底精确匹配）下放宽到 0.3 提升召回，
    # 口语化短查询（如"衣服怎么洗"）更易命中，无关查询仍有 BM25 门槛兜底
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3
    CHAT_HISTORY_WINDOW: int = 10

    # === 上下文压缩（滑动窗口 + 超阈值摘要） ===
    CONTEXT_RECENT_TURNS: int = 4              # 近几轮保持原文高保真
    CONTEXT_COMPRESS_THRESHOLD_TURNS: int = 4  # 历史轮数超过此值才触发摘要压缩

    # === 检索优化（混合检索 / 重排 / 改写 / 缓存） ===
    HYBRID_ENABLED: bool = True        # BM25 + 向量混合检索（RRF 融合）
    HYBRID_RRF_K: int = 60             # RRF 融合常数 k
    BM25_TOP_K: int = 5                # BM25 路召回数
    BM25_TOKENIZER: str = "jieba"      # 中文分词: "jieba" | "char_bigram"（jieba 缺失时的零依赖降级）
    RERANK_ENABLED: bool = False       # 重排开关
    RERANK_MODE: str = "bm25"          # "none" | "bm25" | "api"（api 仅占位，暂未实现）
    RERANK_TOP_N: int = 10             # 重排输入候选数（召回量）
    RERANK_OUTPUT_K: int = 5           # 重排输出数
    RERANK_MODEL: str = "gte-rerank-v2" # API 重排模型（百炼）
    QUERY_REWRITE_ENABLED: bool = False  # 查询改写（结合历史消解指代，多一次 LLM 调用）
    CACHE_ENABLED: bool = False        # 检索结果缓存
    CACHE_TTL: int = 300               # 缓存秒数
    CACHE_MAX_ENTRIES: int = 256

    # === MinerU 在线解析（PDF → Markdown，可选） ===
    # 开关打开且 token 非空时 PDF 走 MinerU；任一条件不满足自动降级 PyPDFLoader。
    # token 在 https://mineru.net 的「API 管理 → Token」创建，有效期 90 天且不支持续期，
    # 到期后需重新创建并更新 .env。
    MINERU_ENABLED: bool = True
    MINERU_API_TOKEN: str = ""
    MINERU_BASE_URL: str = "https://mineru.net/api/v4"
    # 轮询总超时。实测 180 页的费率表约需 5 分钟以上，300 秒不够，
    # 默认给到 600 秒；超时后自动降级 PyPDFLoader，不会让入库失败。
    MINERU_TIMEOUT: int = 600
    MINERU_LANGUAGE: str = "ch"        # OCR 语言，中文场景保持 ch（中英混合）

    # === 文件上传 ===
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = "txt,md,pdf,docx,csv,xlsx"

    # === CORS ===
    CORS_ORIGINS: str = "http://localhost:5175,http://localhost:3000"

    # === Admin 种子数据 ===
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "123456"

    @property
    def allowed_extensions_list(self) -> List[str]:
        # 统一小写：上传侧已 lower，配置里写 "PDF" 也必须能匹配上
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# 全局单例
settings = Settings()
