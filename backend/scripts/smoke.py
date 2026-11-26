"""冒烟验收 —— 一条命令回答「服务起没起来、主链路通不通」

只走公开 HTTP 接口，不碰数据库、不调模型，几十秒内出结果。
适用于每次改完代码、合并前、或者线上出问题时的第一轮排查。

覆盖：
1. 服务存活（/openapi.json 能返回且包含关键路由）
2. 登录拿到 token
3. 拿到文档型知识库
4. 上传一个小 Markdown → 后台入库完成（status = indexed）
5. 检索能命中刚上传的内容（验证向量库与父子块链路）
6. 删除测试文档，不在库里留垃圾

用法（后端已在 :8000 运行）:
    cd backend
    venv\\Scripts\\python.exe scripts/smoke.py [API 地址，默认 http://127.0.0.1:8000]

退出码：全部通过 0，任一失败 1。
"""

import asyncio
import os
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ASKBASE_API", "http://127.0.0.1:8000")
ADMIN_USER = os.environ.get("ASKBASE_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ASKBASE_ADMIN_PASS", "123456")

# 上传的探针内容：唯一标识词用于验证「检索真的命中这一份」
PROBE = "冒烟验收探针"
PROBE_BODY = f"""# {PROBE}

## 南极观测站值班规则

本站代号 {PROBE}，值班人员每四小时轮换一次，交接时必须核对温度记录仪读数。
如遇设备离线，先检查供电，再上报值班长。
"""

POLL_TIMEOUT = 180
POLL_INTERVAL = 3


class Result:
    def __init__(self):
        self.rows = []

    def ok(self, name, detail=""):
        self.rows.append((True, name, detail))
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name, detail=""):
        self.rows.append((False, name, detail))
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

    @property
    def failed(self):
        return [r for r in self.rows if not r[0]]


async def main() -> int:
    r = Result()
    doc_id = None

    async with httpx.AsyncClient(base_url=BASE, timeout=60) as c:
        # 1. 服务存活
        try:
            resp = await c.get("/openapi.json")
            paths = resp.json().get("paths", {})
            need = ["/api/auth/login", "/api/kb/documents/upload", "/api/kb/search"]
            missing = [p for p in need if p not in paths]
            if resp.status_code == 200 and not missing:
                r.ok("服务存活", f"{len(paths)} 个路由")
            else:
                r.fail("服务存活", f"缺少路由 {missing}")
                return 1
        except Exception as e:  # noqa: BLE001 — 连不上就是失败，直接给结论
            r.fail("服务存活", f"连不上 {BASE}: {e}")
            return 1

        # 2. 登录
        try:
            resp = await c.post(
                "/api/auth/login",
                data={"username": ADMIN_USER, "password": ADMIN_PASS},
            )
            if resp.status_code != 200:
                r.fail("登录", f"HTTP {resp.status_code}: {resp.text[:120]}")
                return 1
            token = resp.json()["access_token"]
            c.headers["Authorization"] = f"Bearer {token}"
            me = await c.get("/api/auth/me")
            r.ok("登录", f"{me.json().get('username')} / {me.json().get('role')}")
        except Exception as e:  # noqa: BLE001
            r.fail("登录", str(e))
            return 1

        # 3. 找文档型知识库
        try:
            kbs = (await c.get("/api/knowledge-bases")).json()
            docs_kb = [k for k in kbs if k.get("type") == "document"]
            if not docs_kb:
                r.fail("文档型知识库", "库里没有文档型知识库，无法验证上传链路")
                return 1
            kb_id = docs_kb[0]["id"]
            r.ok("文档型知识库", f"{docs_kb[0]['name']} (id={kb_id})")
        except Exception as e:  # noqa: BLE001
            r.fail("文档型知识库", str(e))
            return 1

        # 4. 上传 → 等入库
        try:
            resp = await c.post(
                "/api/kb/documents/upload",
                files={"file": ("smoke_probe.md", PROBE_BODY.encode("utf-8"), "text/markdown")},
                data={"kb_id": str(kb_id)},
            )
            if resp.status_code != 201:
                r.fail("上传文档", f"HTTP {resp.status_code}: {resp.text[:160]}")
                return 1
            doc_id = resp.json()["id"]

            deadline = time.time() + POLL_TIMEOUT
            status = "processing"
            while time.time() < deadline:
                await asyncio.sleep(POLL_INTERVAL)
                d = (await c.get(f"/api/kb/documents/{doc_id}")).json()
                status = d.get("status")
                if status in ("indexed", "failed"):
                    break

            if status == "indexed":
                r.ok("文档入库", f"doc_id={doc_id}, {d.get('chunk_count')} 块")
            else:
                r.fail("文档入库", f"最终状态 {status}（超时 {POLL_TIMEOUT}s）")
        except Exception as e:  # noqa: BLE001
            r.fail("文档入库", str(e))

        # 5. 检索命中
        try:
            resp = await c.get("/api/kb/search", params={"q": PROBE, "top_k": 3})
            results = resp.json().get("results", [])
            hit = any(PROBE in (x.get("chunk_text") or "") for x in results)
            if hit:
                top = results[0]
                r.ok(
                    "检索命中",
                    f"{len(results)} 条，首位来自 {top.get('filename')}"
                    f"（章节 {top.get('metadata', {}).get('section_path') or '无'}）",
                )
            else:
                r.fail("检索命中", f"返回 {len(results)} 条但没有命中探针内容")
        except Exception as e:  # noqa: BLE001
            r.fail("检索命中", str(e))

        # 6. 清理
        if doc_id is not None:
            try:
                resp = await c.delete(f"/api/kb/documents/{doc_id}")
                if resp.status_code == 200:
                    r.ok("清理测试文档")
                else:
                    r.fail("清理测试文档", f"HTTP {resp.status_code}: {resp.text[:120]}")
            except Exception as e:  # noqa: BLE001
                r.fail("清理测试文档", str(e))

    print()
    if r.failed:
        print(f"冒烟未通过：{len(r.failed)} 项失败")
        return 1
    print(f"冒烟全部通过（{len(r.rows)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
