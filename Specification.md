# Specification — Planner Mode, Sandbox Removal & Ollama Support

**Status:** Draft for review (no code written yet)
**Author:** Pranal Dongare (with Claude Code)
**Date:** 2026-06-08
**Repo:** `huggingface/ml-intern` (branch `main`)

---

## 1. Goal

Pivot ML Intern from an *autonomous executor* into a *planner / advisor*, and make it runnable against local Ollama models. Concretely:

1. **Remove the Hugging Face Sandbox dependency.**
2. **Stop just before execution.** The agent researches and validates (read-only), then emits a **detailed, runnable "How" execution plan (a runbook)** documenting every step it *would* have performed — and stops. No code is executed anywhere (no sandbox, no local shell, no `hf_jobs`).
3. **Add local Ollama model support** (validated as *not present today*).

### Decisions locked in (from review Q&A)

| Decision | Choice |
|---|---|
| How much execution to give up | **Plan only — no execution at all** (no sandbox, no local bash/write/edit, no `hf_jobs`) |
| Toggle vs default | **Permanent default** — planning *is* the product behavior |
| Ollama | **Add support** in this effort (not just validate) |
| Sandbox removal style | **Disable / route-around, keep files** (don't delete `sandbox_*.py`; just stop registering & referencing them) |
| Scope | **Full stack — agent core + CLI + backend + frontend** (demo target: today evening) |
| HF token | **Still required** — used for HF model/dataset API checks. We only remove the **HF Sandbox** and **`hf_jobs`** requirements, not HF connectivity. |
| Runbook output | **One uniquely-named `RUNBOOK` file per run** for easy identification (see §3.1). |

---

## 2. Current-state analysis (verified in code)

### 2.1 The Sandbox is already bypassed in the CLI

The CLI constructs the tool router in **local mode**, which swaps the remote sandbox for local file/shell tools:

- [agent/main.py:854](agent/main.py#L854) and [agent/main.py:1076](agent/main.py#L1076): `ToolRouter(..., local_mode=True)`.
- [agent/core/tools.py:381-385](agent/core/tools.py#L381-L385): `local_mode=True` → `get_local_tools()`; else → `get_sandbox_tools()`.
- Local tools (`bash/read/write/edit` on the real filesystem): [agent/tools/local_tools.py](agent/tools/local_tools.py).
- The **HF Sandbox** is therefore used **only by the web/backend path** — [backend/session_manager.py:169](backend/session_manager.py#L169) builds `ToolRouter(...)` *without* `local_mode`, so it defaults to `False`.

### 2.2 What the Sandbox actually is

[agent/tools/sandbox_client.py](agent/tools/sandbox_client.py) creates a sandbox by **duplicating a template HF Space** (`burtenshaw/sandbox`) and talking to it over HTTPS. Lifecycle, auto-create, orphan-sweeping and trackio-seeding live in [agent/tools/sandbox_tool.py](agent/tools/sandbox_tool.py). Cleanup/cancel hooks: [backend/session_manager.py:633](backend/session_manager.py#L633), [agent/core/agent_loop.py:373](agent/core/agent_loop.py#L373).

### 2.3 `hf_jobs` is a *separate* remote-execution dependency

Real training runs via **HF Jobs cloud** ([agent/tools/jobs_tool.py](agent/tools/jobs_tool.py)), not the sandbox. The sandbox is for iterative dev/testing *before* launching a job. "Plan only" therefore must neutralize **both** the sandbox and `hf_jobs` (and local exec), or the agent would still execute at scale.

### 2.4 Approval gate (the natural "stop point")

[agent/core/agent_loop.py:113-180](agent/core/agent_loop.py#L113-L180) (`_needs_approval`) already classifies the execution-edge tools — `sandbox_create`, `hf_jobs`, `hf_repo_files` (upload/delete), `hf_repo_git` (mutations). This is conceptually the line we are moving the "stop" to: instead of *approving* execution, the agent *plans* it.

### 2.5 Model routing — Ollama is unsupported

[agent/core/llm_params.py:140-201](agent/core/llm_params.py#L140-L201) has exactly four branches:
- `anthropic/…`, `bedrock/…`, `openai/…`, and a **catch-all** that rewrites *any other* id to `openai/<id>` + `api_base=https://router.huggingface.co/v1`.

So `ollama/llama3` would be sent to the **HF router** and fail. There are **zero** `ollama` references in the repo. The CLI also **hard-requires an HF token at startup** ([agent/main.py:824-826](agent/main.py#L824-L826)).

External validation: LiteLLM supports Ollama natively via `ollama/` and (preferred for tools) `ollama_chat/`, requiring api_base (`OLLAMA_API_BASE`, default `http://localhost:11434`). Tool-calling works but is **model-dependent** and can be unreliable on small models (falls back to JSON-mode).

---

## 3. Proposed design

### 3.1 Workstream A — Planner ("stop before execution") toolset

**Principle:** the agent keeps all *read-only research/validation* tools, loses all *execution/mutation* tools, and gains one *terminal deliverable* tool.

**Keep (read-only):**
`research`, `explore_hf_docs`, `hf_docs_fetch`, `hf_papers`, `web_search`, `hf_inspect_dataset`, `github_find_examples`, `github_list_repos`, `github_read_file`, `read` (local read-only — *proposed keep*; see Open Question O3), `plan_tool` (todo/progress tracker), `notify`.

**Remove from registration (route-around, files stay on disk):**
`sandbox_create` + the 4 sandbox ops, `bash`, `write`, `edit`, `hf_jobs`, `hf_repo_files` (upload/delete), `hf_repo_git` (mutations).

**Add — `execution_plan` tool (the "How" runbook):**
The agent's single terminal action. Validates a structured schema, renders Markdown, writes a **uniquely-named runbook file per run**, emits a `plan_ready` event, and ends the turn.

**Runbook file naming:** `runbooks/RUNBOOK-<YYYYMMDD-HHMMSS>-<slug>.md`, where `<slug>` is a short kebab-case slug derived from the objective. Files live in a `runbooks/` directory in the working dir so successive runs never overwrite each other and are easy to identify at a glance. The tool returns the path and the plan is surfaced to the UI via the `plan_ready` event.

Proposed schema (final shape open for review):

```jsonc
{
  "objective": "string — restate the task precisely",
  "assumptions": ["string"],
  "prerequisites": ["env vars, tokens, packages, hardware"],
  "resources": {
    "models":   [{"id": "org/model", "url": "https://huggingface.co/...", "why": "..."}],
    "datasets": [{"id": "org/ds",    "url": "...", "columns_verified": true, "why": "..."}],
    "references":[{"title": "...", "url": "...", "what_we_used": "..."}]
  },
  "steps": [{
    "id": "1",
    "title": "short",
    "rationale": "why this step / what research backs it",
    "actions": "exact commands or code block(s) to run",
    "expected_result": "...",
    "validation": "how to confirm it worked",
    "risks": "what can go wrong + mitigation"
  }],
  "training_config": {            // optional, when ML training is involved
    "method": "SFT|DPO|GRPO|...",
    "hyperparameters": {},
    "hardware": "a10g-large",
    "estimated_time": "~2h",
    "estimated_cost": "approx"
  },
  "evaluation_plan": "how the result would be measured",
  "risks_and_mitigations": ["string"],
  "open_questions": ["string"]
}
```

**Implementation points:**
- New `agent/tools/execution_plan_tool.py` (spec + handler), registered in [agent/core/tools.py](agent/core/tools.py).
- Restructure `create_builtin_tools()` ([agent/core/tools.py:284](agent/core/tools.py#L284)) into the planner toolset; the `local_mode` branch ([:381](agent/core/tools.py#L381)) becomes moot since neither sandbox nor local-exec tools are registered (keep the param for now to minimize churn).
- **System prompt rewrite** ([agent/prompts/system_prompt_v3.yaml](agent/prompts/system_prompt_v3.yaml)): delete "Sandbox-first development" (L123-128), the `hf_jobs` pre-flight/execution sections (L103-121), and the autonomous **"LOOP UNTIL TIME RUNS OUT / NEVER respond with only text / NEVER STOP WORKING"** block (L156-182) — these directly contradict "stop and hand off a plan." Replace with planner instructions: *research deeply (read-only) → produce one complete `execution_plan` → stop.* Keep the literature/research-first workflow, data-audit, and Trackio *guidance* (as plan content, not as executed steps).
- Adjust the local-mode prompt addendum ([agent/context_manager/manager.py:196-209](agent/context_manager/manager.py#L196-L209)) to describe planner mode instead of "run code directly with bash."
- The agentic loop needs **no structural change** to "stop": a turn ends naturally when the assistant emits no tool calls. Removing the "never stop" prompt + making `execution_plan` the final tool is sufficient.

### 3.2 Workstream B — Remove the HF Sandbox dependency (route-around)

- Sandbox tools are simply **not registered** (covered by A).
- Backend ([backend/session_manager.py:169](backend/session_manager.py#L169)): set `local_mode=True` (or, cleaner, drop the sandbox branch) so the web path matches the CLI. Cleanup/cancel sandbox hooks ([:633](backend/session_manager.py#L633), [agent_loop.py:373](agent/core/agent_loop.py#L373)) become no-ops since `session.sandbox` is never set — leave guarded code in place.
- `hf_jobs` script path-resolution that reads from the sandbox ([agent_loop.py:1173-1179](agent/core/agent_loop.py#L1173)) is removed along with `hf_jobs`.
- **Files kept on disk** (per decision): `sandbox_tool.py`, `sandbox_client.py`, `trackio_seed.py`, `scripts/sweep_orphan_sandboxes.py`. They become unreferenced but available for revert.
- **Frontend / web:** out of primary scope but affected — sandbox & jobs UI ([frontend/src/components/Chat/ToolCallGroup.tsx](frontend/src/components/Chat/ToolCallGroup.tsx), `ActivityStatusBar.tsx`, `JobsUpgradeDialog.tsx`, billing/credits flows) would render for tools that no longer fire. See Open Question O1.

### 3.3 Workstream C — Add Ollama support

- **New branch in `_resolve_llm_params`** ([agent/core/llm_params.py](agent/core/llm_params.py)): if `model_name` starts with `ollama/` or `ollama_chat/`, return `{"model": <id>, "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434")}`. Prefer normalizing `ollama/` → `ollama_chat/` for better tool-calling. Drop `reasoning_effort` (most local models don't support it); add an empty Ollama effort set for `strict` mode.
- **Model validation/switch** ([agent/core/model_switcher.py](agent/core/model_switcher.py)): short-circuit `ollama*` ids in `_print_hf_routing_info` (like `anthropic/`/`openai/`) so we don't query the HF catalog; the effort probe already degrades gracefully via `ProbeInconclusive`. Optionally add an example to `SUGGESTED_MODELS`.
- **Relax the HF-token hard block** ([agent/main.py:824-826](agent/main.py#L824-L826)): when the active model is `ollama*`, allow continuing without an HF token (warn that HF research tools will be limited). See O2.
- **Context window:** local models often expose 8k–32k context. The compaction threshold is derived from `litellm.get_model_info` ([manager.py:152-156](agent/context_manager/manager.py#L152)); verify it resolves a sane value for Ollama ids and falls back safely if unknown.
- **Usage:** `ml-intern --model ollama_chat/llama3.1` (or `/model ollama_chat/qwen2.5`), with Ollama installed and serving. Document `OLLAMA_API_BASE` in the README.

---

## 4. Risks & limitations

1. **"Local/offline" is only partial.** Ollama swaps the *reasoning LLM* to local, but the **research tools still call Hugging Face/GitHub/web over the network and need an HF token**. True offline operation is out of scope.
2. **Tool-calling reliability on small Ollama models.** The agent is tool-call-heavy; small models may fail to emit valid tool calls or hallucinate args, stalling the loop. Recommend tool-capable models (llama3.1, qwen2.5-coder, etc.) and document this.
3. **Context limits.** Small local context windows can trigger frequent compaction or `ContextWindowExceededError`; the research-heavy workflow produces large contexts. Mitigation: verify the model-info-driven threshold; possibly cap research breadth in planner mode.
4. **Web/backend & frontend drift.** Removing sandbox + jobs leaves dead UI (sandbox panels, jobs billing/credits). If web is in scope, this is a larger, cross-stack change (O1).
5. **Plan quality is unverified by construction.** Because nothing runs, the runbook can contain APIs/commands that wouldn't actually work — the whole point of the sandbox was to catch these. The system prompt must lean hard on research/validation, and the runbook should mark unverified assumptions explicitly.
6. **Tests & telemetry.** Sandbox/jobs unit + integration tests ([tests/unit/test_sandbox_*.py](tests/unit), [tests/integration/test_live_sandbox_auth.py](tests/integration)) and telemetry events (`sandbox_create`, jobs KPIs) become obsolete/dormant; they need updating or skipping. New tests required for `execution_plan` and the Ollama routing branch.
7. **Session persistence / trajectory dumps** assume tool-execution traces; planner traces differ but should still serialize fine (no schema change expected).

---

## 5. Open questions — RESOLVED

- **O1 — Scope:** ✅ **Full stack** — agent core + CLI + backend + frontend, in this pass (demo today evening).
- **O2 — HF token:** ✅ **Keep required.** It's used for HF model/dataset API checks. We remove only the HF Sandbox + `hf_jobs` requirements.
- **O3 — Keep read-only `read`:** ✅ **Keep `read`, drop `bash/write/edit`.**
- **O4 — Deliverable form:** ✅ **Structured `execution_plan` tool** that writes a runbook file.
- **O5 — Naming:** ✅ Tool name `execution_plan`; output `runbooks/RUNBOOK-<YYYYMMDD-HHMMSS>-<slug>.md` (one per run).

---

## 6. Rough implementation order (once approved)

1. Add `execution_plan` tool + schema + Markdown renderer/writer.
2. Rework `create_builtin_tools()` to the planner toolset; stop registering sandbox/exec/jobs tools.
3. Rewrite `system_prompt_v3.yaml` (+ local-mode addendum) for planner behavior.
4. Backend route-around (`local_mode=True`); keep sandbox files dormant.
5. Add Ollama branch to `llm_params.py` + `model_switcher.py`; relax HF-token block.
6. Update/skip obsolete tests; add tests for `execution_plan` and Ollama routing.
7. README updates (Ollama usage, planner-mode description); remove sandbox-first docs.

---

## 7. Out of scope (this pass)

- Deleting sandbox/jobs source files (kept dormant by decision).
- Full frontend retirement of sandbox/jobs/billing UI (pending O1).
- Offline mode for research tools.
- Auto-executing or self-verifying the generated runbook.
