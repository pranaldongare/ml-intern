"""
execution_plan tool — the planner's single terminal deliverable.

ML Intern runs in *planner mode*: it researches and validates using read-only
tools, then instead of executing anything (no sandbox, no local shell, no
hf_jobs) it emits a detailed, runnable "How" plan — a runbook — and stops.

This tool:
  * validates a structured execution-plan schema,
  * renders it to Markdown,
  * writes a uniquely-named runbook file per run
    (``runbooks/RUNBOOK-<YYYYMMDD-HHMMSS>-<slug>.md``),
  * emits a ``plan_ready`` event so the UI can surface the runbook,
  * returns the path so the agent can reference it in its closing message.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.core.session import Event

logger = logging.getLogger(__name__)

RUNBOOKS_DIR = "runbooks"
_SLUG_MAX_LEN = 48


# ── helpers ───────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Short kebab-case slug for filenames. Falls back to 'plan'."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if not text:
        return "plan"
    return text[:_SLUG_MAX_LEN].strip("-")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _md_bullets(items: list, empty: str = "_None._") -> str:
    items = [str(i).strip() for i in _as_list(items) if str(i).strip()]
    if not items:
        return empty
    return "\n".join(f"- {i}" for i in items)


def _render_resources(resources: dict[str, Any]) -> str:
    if not isinstance(resources, dict) or not resources:
        return "_None specified._"
    # Accept both plural array form ({"models": [...]}) and singular/string
    # form ({"model": "...", "dataset": "..."}) that weaker models emit.
    _ALIASES = {
        "models": ("models", "model"),
        "datasets": ("datasets", "dataset"),
        "references": ("references", "reference", "refs", "docs"),
    }
    lines: list[str] = []
    rendered_any = False
    for kind, keys in _ALIASES.items():
        entries: list = []
        for k in keys:
            entries.extend(_as_list(resources.get(k)))
        if not entries:
            continue
        lines.append(f"\n**{kind.capitalize()}:**")
        for e in entries:
            if isinstance(e, dict):
                label = e.get("id") or e.get("name") or e.get("title") or "(unnamed)"
                url = e.get("url")
                why = e.get("why") or e.get("what_we_used") or ""
                verified = e.get("columns_verified")
                bits = [f"`{label}`"]
                if url:
                    bits.append(f"<{url}>")
                if verified is True:
                    bits.append("(columns verified)")
                if why:
                    bits.append(f"— {why}")
                lines.append(f"- {' '.join(bits)}")
            else:
                lines.append(f"- {e}")
        rendered_any = True
    # Surface any leftover keys (e.g. {"trl_docs": "..."}) we didn't map above.
    leftover = {
        k: v for k, v in resources.items()
        if k not in {a for keys in _ALIASES.values() for a in keys}
    }
    if leftover:
        lines.append("\n**Other:**")
        for k, v in leftover.items():
            lines.append(f"- {k}: {v}")
        rendered_any = True
    return "\n".join(lines) if rendered_any else "_None specified._"


def _render_action_block(value: Any) -> str:
    """Render a step's action/code body. If the model already wrapped it in a
    Markdown code fence, keep it as-is; otherwise wrap it in a bash fence."""
    text = str(value).strip()
    if not text:
        return ""
    if "```" in text:
        return f"{text}\n"
    return f"```bash\n{text}\n```\n"


def _render_steps(steps: list) -> str:
    steps = _as_list(steps)
    if not steps:
        return "_No steps provided._"
    out: list[str] = []
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            out.append(f"### Step {idx}\n\n{step}\n")
            continue
        sid = step.get("id") or step.get("step") or str(idx)
        title = str(step.get("title", "")).strip() or "(untitled)"
        out.append(f"### Step {sid} — {title}\n")
        # rationale ← rationale | description (weaker models use "description")
        rationale = step.get("rationale") or step.get("description")
        if rationale:
            out.append(f"**Why:** {rationale}\n")
        # actions ← actions | code | command | code_change
        action = (
            step.get("actions")
            or step.get("code")
            or step.get("command")
            or step.get("code_change")
        )
        if action:
            block = _render_action_block(action)
            if block:
                out.append(f"**Actions:**\n\n{block}")
        if step.get("expected_result"):
            out.append(f"**Expected result:** {step['expected_result']}\n")
        if step.get("validation"):
            out.append(f"**Validation:** {step['validation']}\n")
        if step.get("risks"):
            risks = step["risks"]
            risks = "; ".join(str(r) for r in risks) if isinstance(risks, list) else risks
            out.append(f"**Risks:** {risks}\n")
        out.append("")
    return "\n".join(out)


def _render_training_config(tc: dict[str, Any]) -> str:
    if not isinstance(tc, dict) or not tc:
        return ""
    lines = ["\n## Training configuration\n"]
    simple = {k: v for k, v in tc.items() if k != "hyperparameters"}
    for k, v in simple.items():
        lines.append(f"- **{k.replace('_', ' ').capitalize()}:** {v}")
    hp = tc.get("hyperparameters")
    if isinstance(hp, dict) and hp:
        lines.append("- **Hyperparameters:**")
        for k, v in hp.items():
            lines.append(f"    - `{k}` = `{v}`")
    return "\n".join(lines) + "\n"


def render_markdown(plan: dict[str, Any]) -> str:
    """Render a validated plan dict to a Markdown runbook."""
    objective = plan.get("objective", "").strip() or "(no objective stated)"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts: list[str] = []
    parts.append(f"# Runbook — {objective}\n")
    parts.append(f"_Generated by ML Intern (planner mode) at {generated}._\n")
    parts.append(
        "> This is a **plan only**. Nothing has been executed. Review every "
        "step, verify any assumptions marked below, and run it yourself.\n"
    )

    parts.append("## Objective\n")
    parts.append(objective + "\n")

    parts.append("## Assumptions\n")
    parts.append(_md_bullets(plan.get("assumptions")) + "\n")

    parts.append("## Prerequisites\n")
    parts.append(_md_bullets(plan.get("prerequisites")) + "\n")

    parts.append("## Resources\n")
    parts.append(_render_resources(plan.get("resources", {})) + "\n")

    parts.append("## Steps\n")
    parts.append(_render_steps(plan.get("steps", [])))

    tc = _render_training_config(plan.get("training_config", {}))
    if tc:
        parts.append(tc)

    if plan.get("evaluation_plan"):
        parts.append("## Evaluation plan\n")
        parts.append(str(plan["evaluation_plan"]) + "\n")

    parts.append("## Risks & mitigations\n")
    parts.append(_md_bullets(plan.get("risks_and_mitigations")) + "\n")

    if plan.get("open_questions"):
        parts.append("## Open questions\n")
        parts.append(_md_bullets(plan.get("open_questions")) + "\n")

    return "\n".join(parts).strip() + "\n"


# ── PDF rendering ─────────────────────────────────────────────────────

# Core PDF fonts are latin-1; map the few unicode chars our markdown uses to
# safe ASCII so rendering never errors on them.
_PDF_CHAR_MAP = {
    "—": "-", "–": "-",          # em / en dash
    "→": "->", "←": "<-",        # arrows
    "•": "*",                          # bullet
    "‘": "'", "’": "'",          # smart single quotes
    "“": '"', "”": '"',          # smart double quotes
    "…": "...",                        # ellipsis
    "✅": "[x]", "\U0001f4cb": "",     # check mark, clipboard emoji
}


def _latin1(text: str) -> str:
    for uni, ascii_ in _PDF_CHAR_MAP.items():
        text = text.replace(uni, ascii_)
    return text.encode("latin-1", "replace").decode("latin-1")


def _render_pdf(markdown_text: str, pdf_path: Path) -> bool:
    """Best-effort: render the markdown runbook to a PDF at *pdf_path*.

    Uses fpdf2 (pure-Python, no native deps). Parses the runbook's own
    markdown (headings, blockquote, bullets, fenced code) — we control that
    format, so a light line parser is sufficient. Returns True on success;
    never raises, so a PDF problem can't break the turn or the .md output.
    """
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except Exception as e:
        logger.info("[plan-trace] PDF skipped (fpdf2 not installed: %s)", e)
        return False
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)

        def _cell(text: str, h: float, **kw) -> None:
            # Always reset the cursor to the left margin and advance down, so
            # each line gets the full page width (avoids fpdf2's
            # "Not enough horizontal space" when the cursor stays at the right).
            pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kw)

        in_code = False
        code_buf: list[str] = []

        def _flush_code() -> None:
            if not code_buf:
                return
            pdf.set_font("Courier", size=8)
            pdf.set_fill_color(244, 244, 244)
            for cl in code_buf:
                _cell(cl if cl else " ", 4.5, fill=True)
            pdf.set_font("Helvetica", size=11)
            pdf.ln(2)
            code_buf.clear()

        for raw in markdown_text.split("\n"):
            line = _latin1(raw.rstrip())
            if line.strip().startswith("```"):
                if in_code:
                    _flush_code()
                in_code = not in_code
                continue
            if in_code:
                code_buf.append(line)
                continue

            stripped = line.strip()
            if not stripped:
                pdf.ln(3)
            elif stripped.startswith("### "):
                pdf.set_font("Helvetica", "B", 12); _cell(stripped[4:], 6); pdf.set_font("Helvetica", size=11); pdf.ln(1)
            elif stripped.startswith("## "):
                pdf.set_font("Helvetica", "B", 14); _cell(stripped[3:], 7); pdf.set_font("Helvetica", size=11); pdf.ln(1)
            elif stripped.startswith("# "):
                pdf.set_font("Helvetica", "B", 18); _cell(stripped[2:], 9); pdf.set_font("Helvetica", size=11); pdf.ln(2)
            elif stripped.startswith("> "):
                pdf.set_text_color(110); pdf.set_font("Helvetica", "I", 10); _cell(stripped[2:], 5); pdf.set_text_color(0); pdf.set_font("Helvetica", size=11)
            elif stripped[:2] in ("- ", "* "):
                _cell("  - " + stripped[2:], 5, markdown=True)
            else:
                _cell(line, 5, markdown=True)

        _flush_code()
        pdf.output(str(pdf_path))
        return True
    except Exception as e:
        logger.warning("[plan-trace] PDF generation failed: %s", e)
        return False


# ── tool spec ─────────────────────────────────────────────────────────

EXECUTION_PLAN_TOOL_SPEC = {
    "name": "execution_plan",
    "description": (
        "Emit the FINAL execution plan (runbook) for the user's task and finish.\n\n"
        "ML Intern is a PLANNER: you research and validate with read-only tools, then "
        "call this tool exactly once to hand off a complete, runnable plan. You do NOT "
        "execute anything — there is no sandbox, no shell, and no job runner.\n\n"
        "The plan must be detailed enough that a human (or another agent) could run it "
        "verbatim: exact commands/code, the datasets/models to use (with Hub URLs you "
        "verified), hyperparameters, validation checks, and risks. Mark any assumption "
        "you could not verify. Call this only after you have done the research."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["objective", "steps"],
        "properties": {
            "objective": {
                "type": "string",
                "description": "Restate the user's task precisely.",
            },
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Assumptions made, especially any you could NOT verify.",
            },
            "prerequisites": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Env vars, tokens, packages, hardware needed before running.",
            },
            "resources": {
                "type": "object",
                "additionalProperties": True,
                "description": "Models/datasets/references to use, each ideally with a Hub URL.",
                "properties": {
                    "models": {"type": "array", "items": {"type": "object"}},
                    "datasets": {"type": "array", "items": {"type": "object"}},
                    "references": {"type": "array", "items": {"type": "object"}},
                },
            },
            "steps": {
                "type": "array",
                "minItems": 1,
                "description": "Ordered steps. Each step is concrete and runnable.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "actions"],
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "rationale": {"type": "string", "description": "Why this step / research backing."},
                        "actions": {"type": "string", "description": "Exact commands or code to run."},
                        "expected_result": {"type": "string"},
                        "validation": {"type": "string", "description": "How to confirm success."},
                        "risks": {"type": "string", "description": "What can go wrong + mitigation."},
                    },
                },
            },
            "training_config": {
                "type": "object",
                "additionalProperties": True,
                "description": "Optional. method, hyperparameters, hardware, estimated_time, estimated_cost.",
            },
            "evaluation_plan": {
                "type": "string",
                "description": "How the result would be measured.",
            },
            "risks_and_mitigations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "open_questions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
}


async def execution_plan_handler(
    arguments: dict[str, Any], session: Any = None, tool_call_id: str | None = None
) -> tuple[str, bool]:
    logger.info(
        "[plan-trace] execution_plan handler CALLED: arg_keys=%s objective_len=%d steps=%d",
        list(arguments.keys()),
        len((arguments.get("objective") or "").strip()),
        len(_as_list(arguments.get("steps"))),
    )
    objective = (arguments.get("objective") or "").strip()
    steps = _as_list(arguments.get("steps"))
    if not objective:
        logger.info("[plan-trace] execution_plan REJECTED: missing objective")
        return "Error: 'objective' is required. Re-call execution_plan with a clear objective.", False
    if not steps:
        logger.info("[plan-trace] execution_plan REJECTED: missing steps")
        return "Error: 'steps' must contain at least one step. Re-call with concrete steps.", False

    markdown = render_markdown(arguments)

    # Unique, identifiable filename per run.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slugify(objective)
    filename = f"RUNBOOK-{ts}-{slug}.md"
    pdf_filename = f"RUNBOOK-{ts}-{slug}.pdf"

    path_str: str | None = None
    pdf_path_str: str | None = None
    write_error: str | None = None
    try:
        out_dir = Path(os.getcwd()) / RUNBOOKS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        out_path.write_text(markdown, encoding="utf-8")
        path_str = str(out_path)
        logger.info("[plan-trace] RUNBOOK written: %s (%d bytes)", path_str, len(markdown))
        # Also save a PDF alongside the .md (best-effort — never fail the turn).
        pdf_path = out_dir / pdf_filename
        if _render_pdf(markdown, pdf_path):
            pdf_path_str = str(pdf_path)
            logger.info("[plan-trace] RUNBOOK PDF written: %s", pdf_path_str)
    except Exception as e:  # never fail the turn on a write error — UI still gets the plan
        write_error = str(e)
        logger.warning("[plan-trace] RUNBOOK write FAILED in %s: %s", os.getcwd(), e)

    # Surface the runbook to the UI (web + CLI handlers can render this).
    if session is not None:
        try:
            await session.send_event(
                Event(
                    event_type="plan_ready",
                    data={
                        "tool_call_id": tool_call_id,
                        "objective": objective,
                        "path": path_str,
                        "pdf_path": pdf_path_str,
                        "filename": filename,
                        "pdf_filename": pdf_filename if pdf_path_str else None,
                        "markdown": markdown,
                    },
                )
            )
        except Exception:
            pass

    if path_str:
        pdf_note = f"\nPDF: {pdf_path_str}" if pdf_path_str else ""
        return (
            f"Execution plan written to {path_str}{pdf_note}\n\n"
            f"This is a plan only — nothing was executed. "
            f"Review the runbook and run it yourself.\n\n"
            f"{markdown}"
        ), True

    return (
        f"Execution plan generated (could not write file: {write_error}).\n\n{markdown}"
    ), True
