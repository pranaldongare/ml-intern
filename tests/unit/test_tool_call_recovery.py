"""Regression tests for recovering tool calls that local models (Ollama/qwen)
emit as JSON in the message *content* instead of the structured tool_calls
field. Each payload here is a real shape observed in the field.
"""

import json

import pytest

from agent.core.agent_loop import _recover_tool_calls_from_content
from agent.tools.plan_tool import plan_tool_handler, get_current_plan

KNOWN = {
    "plan_tool", "execution_plan", "research", "explore_hf_docs",
    "fetch_hf_docs", "hf_papers", "web_search", "hf_inspect_dataset",
    "notify", "github_find_examples", "github_list_repos",
    "github_read_file", "read",
}


def _names(content: str):
    return [tc.function.name for tc in _recover_tool_calls_from_content(content, KNOWN)]


# ── Exact field-observed payloads ─────────────────────────────────────

def test_recover_tool_calls_as_list_execution_plan():
    payload = json.dumps({"tool_calls": [
        {"name": "execution_plan", "arguments": {"objective": "X", "steps": [{"title": "s", "actions": "e"}]}}
    ]})
    assert _names(payload) == ["execution_plan"]


def test_recover_tool_calls_as_single_object_plan_tool():
    # tool_calls is an OBJECT, not a list (observed with qwen via Ollama).
    payload = json.dumps({"tool_calls": {
        "name": "plan_tool",
        "arguments": {"todos": [
            {"status": "in_progress", "content": "Research NER"},
            {"status": "pending", "content": "Inspect dataset"},
        ]},
    }})
    assert _names(payload) == ["plan_tool"]


def test_recover_from_think_wrapped_content():
    payload = json.dumps({"tool_calls": {"name": "plan_tool", "arguments": {"todos": []}}})
    wrapped = "<think>I should plan first.</think>\n" + payload
    assert _names(wrapped) == ["plan_tool"]


def test_recover_from_prose_wrapped_content():
    payload = json.dumps({"tool_calls": {"name": "plan_tool", "arguments": {"todos": []}}})
    assert _names("Here is my plan:\n" + payload + "\nProceeding.") == ["plan_tool"]


def test_recover_from_json_fence():
    payload = json.dumps({"tool_calls": [{"name": "research", "arguments": {"task": "t", "context": "c"}}]})
    assert _names("```json\n" + payload + "\n```") == ["research"]


# ── False-positive guards ─────────────────────────────────────────────

def test_no_recovery_from_plain_prose():
    assert _names("Let me research NER approaches before planning.") == []


def test_no_recovery_for_unknown_tool():
    assert _names(json.dumps({"tool_calls": {"name": "do_training", "arguments": {}}})) == []


def test_no_recovery_from_schema_echoed_in_think():
    # Model echoing the execution_plan schema while reasoning must NOT trigger.
    content = (
        "<think>The execution_plan schema is "
        '{"name":"execution_plan","arguments":{"objective":"x","steps":[]}}'
        "</think> I will research first."
    )
    assert _names(content) == []


# ── plan_tool tolerance for id-less / status-less todos ───────────────

@pytest.mark.asyncio
async def test_plan_tool_normalizes_missing_id_and_status():
    args = {"todos": [
        {"content": "Research NER", "status": "in_progress"},
        {"content": "Inspect dataset"},  # no status
    ]}
    out, ok = await plan_tool_handler(args)
    assert ok, out
    plan = get_current_plan()
    assert [t["id"] for t in plan] == ["1", "2"]
    assert plan[0]["status"] == "in_progress"
    assert plan[1]["status"] == "pending"  # defaulted
