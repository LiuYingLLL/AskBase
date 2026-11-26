"""
MinerU 在线解析 — PDF 转结构化 Markdown

调用 MinerU 精准解析 API（https://mineru.net/api/v4）：
    申请上传链接 → PUT 文件二进制 → 轮询 batch 结果 → 下载 zip 取 full.md

设计约束：
- 只做解析，不做切分。返回 Markdown 原文，交给调用方的切分逻辑。
- 任何失败都抛 MinerUError，由调用方决定降级到本地 loader。
- token 从 settings 读取，不落代码、不落日志。
"""

import asyncio
import io
import os
import time
import zipfile

import httpx

from app.config import settings

# 轮询退避：起步 2 秒，每次乘 1.5，上限 10 秒
_POLL_INITIAL_INTERVAL = 2.0
_POLL_MAX_INTERVAL = 10.0


class MinerUError(Exception):
    """MinerU 解析失败，调用方应降级到本地解析"""


def is_mineru_configured() -> bool:
    """开关打开且 token 非空时才认为可用"""
    return bool(settings.MINERU_ENABLED and settings.MINERU_API_TOKEN.strip())


def _base_url() -> str:
    return settings.MINERU_BASE_URL.rstrip("/")


def _json_headers() -> dict:
    """带鉴权的 JSON 请求头（token 不打印、不外泄）"""
    return {
        "Authorization": f"Bearer {settings.MINERU_API_TOKEN.strip()}",
        "Content-Type": "application/json",
    }


async def _request_upload_url(client: httpx.AsyncClient, filename: str) -> tuple[str, str]:
    """申请上传链接，返回 (batch_id, upload_url)"""
    resp = await client.post(
        f"{_base_url()}/file-urls/batch",
        headers=_json_headers(),
        json={
            "enable_formula": True,
            "language": settings.MINERU_LANGUAGE,
            "files": [{"name": filename, "is_ocr": False}],
        },
    )
    if resp.status_code != 200:
        raise MinerUError(f"申请上传链接失败: HTTP {resp.status_code}")

    body = resp.json()
    if body.get("code") != 0:
        raise MinerUError(f"申请上传链接失败: {body.get('msg') or body.get('code')}")

    data = body.get("data") or {}
    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls") or []
    if not batch_id or not file_urls:
        raise MinerUError("申请上传链接失败: 响应缺少 batch_id 或 file_urls")

    return batch_id, file_urls[0]


async def _upload_file(client: httpx.AsyncClient, upload_url: str, file_path: str) -> None:
    """把文件二进制 PUT 到预签名地址

    注意：这里刻意不设置 Content-Type —— 预签名 URL 对请求头敏感，
    多带一个 Content-Type 会导致签名校验失败。
    """
    with open(file_path, "rb") as f:
        content = f.read()

    resp = await client.put(upload_url, content=content)
    if resp.status_code not in (200, 201, 204):
        raise MinerUError(f"上传文件失败: HTTP {resp.status_code}")


async def _poll_result(client: httpx.AsyncClient, batch_id: str) -> str:
    """轮询解析结果，返回 full_zip_url；超时或失败抛 MinerUError"""
    url = f"{_base_url()}/extract-results/batch/{batch_id}"
    deadline = time.monotonic() + settings.MINERU_TIMEOUT
    interval = _POLL_INITIAL_INTERVAL

    while True:
        if time.monotonic() > deadline:
            raise MinerUError(f"轮询超时（{settings.MINERU_TIMEOUT}s）")

        resp = await client.get(url, headers=_json_headers())
        if resp.status_code != 200:
            raise MinerUError(f"查询解析结果失败: HTTP {resp.status_code}")

        body = resp.json()
        if body.get("code") != 0:
            raise MinerUError(f"查询解析结果失败: {body.get('msg') or body.get('code')}")

        results = (body.get("data") or {}).get("extract_result") or []
        if results:
            item = results[0]
            state = item.get("state")
            if state == "done":
                zip_url = item.get("full_zip_url")
                if not zip_url:
                    raise MinerUError("解析完成但未返回 full_zip_url")
                return zip_url
            if state == "failed":
                raise MinerUError(f"解析失败: {item.get('err_msg') or '未知原因'}")

        await asyncio.sleep(interval)
        interval = min(interval * 1.5, _POLL_MAX_INTERVAL)


async def _download_markdown(client: httpx.AsyncClient, zip_url: str) -> str:
    """下载结果 zip 并取出 Markdown 正文"""
    resp = await client.get(zip_url)
    if resp.status_code != 200:
        raise MinerUError(f"下载解析结果失败: HTTP {resp.status_code}")

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = [
                n
                for n in zf.namelist()
                if n.endswith(".md") and not n.startswith("__MACOSX")
            ]
            if not names:
                raise MinerUError("解析结果中没有 Markdown 文件")
            # 优先取 full.md，取不到就退而取第一个 .md
            target = next((n for n in names if n.endswith("full.md")), names[0])
            return zf.read(target).decode("utf-8", errors="replace")
    except zipfile.BadZipFile as e:
        raise MinerUError(f"解析结果不是有效的 zip: {e}") from e


async def parse_pdf_with_mineru(file_path: str) -> str:
    """用 MinerU 在线 API 把 PDF 解析成 Markdown

    成功返回 Markdown 正文；任何异常统一包装成 MinerUError，供调用方降级。
    """
    if not is_mineru_configured():
        raise MinerUError("MinerU 未配置（开关关闭或 token 为空）")

    filename = os.path.basename(file_path)
    timeout = httpx.Timeout(settings.MINERU_TIMEOUT, connect=30.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            batch_id, upload_url = await _request_upload_url(client, filename)
            await _upload_file(client, upload_url, file_path)
            zip_url = await _poll_result(client, batch_id)
            markdown = await _download_markdown(client, zip_url)
    except MinerUError:
        raise
    except httpx.HTTPError as e:
        raise MinerUError(f"网络请求失败: {e}") from e
    except Exception as e:  # noqa: BLE001 — 兜底，保证调用方一定能降级
        raise MinerUError(f"解析异常: {e}") from e

    if not markdown.strip():
        raise MinerUError("解析结果为空")

    return markdown
