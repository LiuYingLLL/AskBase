## Working context

- Before exploring a domain, read root `CONTEXT.md` and relevant `docs/adr/`; follow `docs/agents/domain.md`. The glossary distinguishes conversation agents from crawler pipeline stages.
- Issues and specs live under `.scratch/<feature>/`; use `docs/agents/issue-tracker.md`. Triage role labels follow `docs/agents/triage-labels.md` unchanged.

## Commands and tests

- Run backend commands from `backend/` with `venv\Scripts\python.exe`, not the system Python. Start with `venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --reload`.
- Backend focused test: `venv\Scripts\python.exe -m pytest tests/<file>.py -q`; select one case with `::TestClass::test_name` or `-k <expression>`.
- Backend suite without load/stress runs: `venv\Scripts\python.exe -m pytest tests/ -q --ignore=tests/load -m "not stress"`. Stress tests can call a running backend; skipped tests are not acceptance evidence.
- From `frontend/`: `npm run test -- src/<path>.test.tsx` for focused tests; `npm run build` runs TypeScript project checks before Vite. There is no lint script in its package manifest.
- Frontend tests use the separate `vitest.config.ts`, including `src/test-setup.ts` browser mocks; keep these when changing test configuration.
- Backend DB tests have an autouse usage-write isolation fixture in `tests/conftest.py`. Reuse `temp_db` or `usage_db` for cross-session assertions; these use temporary file SQLite databases.
- Frontend dev port is 5175. Set `ASKBASE_API_PORT` before starting Vite to change its `/api` proxy target (default 8000).

## Test cadence

Full backend suite is ~25s; a focused file is seconds. Spend the difference deliberately.

- **While editing**: run only the test file(s) the change touches, e.g. `pytest tests/test_parent_child.py -q`. Never run the full suite after every small edit.
- **Run the full suite once** when a batch of edits is complete, before handing off or merging — not per file. Use the full backend command listed above.
- **Frontend**: `npx tsc -b` after any `.ts`/`.tsx` change; `npm run test -- src/<path>` for the touched test file. `npm run build` only when the built bundle is itself the deliverable — Vite's `dist` cleanup can be blocked by the sandbox, which is not a code failure.
- **`tests/load` and `-m stress`** are on demand only (they can hit a running backend). Never wire them into the routine loop.
- **Smoke check** (after the backend is up): `venv\Scripts\python.exe scripts/smoke.py`. It goes over real HTTP — liveness, login, upload, indexing, retrieval — and cleans up after itself. Use it as the first step when "is it working?" is the question, before digging into unit tests. Set `NO_PROXY=127.0.0.1` if a system proxy intercepts localhost.
- A change touching `tests/conftest.py`, shared fixtures, or `app/config.py` is wide-reaching: run the full suite.

## Test quality bar

Keep tests that guard behaviour; delete ones that only restate a literal.

- Do **not** assert config literals (`CHUNK_SIZE == 800`) or host environment (`LLM_API_KEY != ""`). The first breaks on every legitimate default change and verifies nothing; the second fails on any other machine.
- Prefer seam-level contracts: parsing, ordering, degradation paths, and invariants that would break silently in production (`CHUNK_OVERLAP < CHUNK_SIZE`, source-number ordering, "a missing link never becomes a fabricated URL").
- When a test needs real DB writes, use the `temp_db` / `usage_db` fixtures. They copy a session-scoped schema template — do not reintroduce per-test `create_all`.

## Wiring and gotchas

- `backend/app/main.py` lifespan performs compatibility ALTERs, seeds built-in tools/models and starts the crawler scheduler. `scripts/init_all.py` only creates missing tables and seeds the admin; it is not a substitute for startup migrations. New ORM fields alone do not migrate existing tables.
- Application Skills are executable tools resolved through `backend/app/skills/registry.py` and `handlers.py`, not OpenCode `SKILL.md` instructions. A database Skill row alone does not supply an executable handler.
- MCP already has admin CRUD, discovery and test-call endpoints in `backend/app/api/mcp_servers.py`. The current UI is read-only; reuse the backend rather than assuming the whole integration is missing.
- MCP runtime currently handles stdio and SSE in `backend/app/services/mcp_service.py`; stdio starts on the backend machine, not the browser. Tool discovery populates a cache used by the agent editor; reading the cached list does not discover tools.
- `frontend/src/pages/AgentEditPage.tsx` binds individual tools using `skill:<id>` or `mcp:<server_id>:<tool_name>`; preserve the distinction between adding a tool/service and attaching it to an agent.
- For tool-output changes, preserve the source-number and tool-call-history contracts in `CONTEXT.md`; plain-text MCP output does not itself provide structured citation sources.
- Vector storage is Chroma with a global singleton client. If retrieval returns nothing while documents clearly exist, check the server log for `Cannot return the results in a contigious 2D array` — that means the hnsw index is corrupt, not that there is no matching content. `probe_index_health()` detects it and `rebuild_index` recreates the collection, so "rebuild index" in the KB page is the repair path. Note `retrieve_with_scores` swallows errors into an empty list, so an empty result never proves the index is fine.
- `chromadb==0.5.23` declares `posthog>=2.4` with no upper bound. posthog 4+ changed `capture()` to keyword-only args, which makes every Chroma collection operation log `capture() takes 1 positional argument but 3 were given`. Keep `posthog<4` pinned; `anonymized_telemetry=False` on the client alone does not stop it, because the flag is read from chromadb's global `system.settings` when the client is constructed.
