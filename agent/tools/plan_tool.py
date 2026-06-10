from typing import Any, Dict, List

from agent.core.session import Event
from agent.utils.terminal_display import format_plan_tool_output

from .types import ToolResult

# In-memory storage for the current plan (raw structure from agent)
_current_plan: List[Dict[str, str]] = []


class PlanTool:
    """Tool for managing a list of todos with status tracking."""

    def __init__(self, session: Any = None):
        self.session = session

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """
        Execute the WritePlan operation.

        Args:
            params: Dictionary containing:
                - todos: List of todo items, each with id, content, and status

        Returns:
            ToolResult with formatted output
        """
        global _current_plan

        todos = params.get("todos", [])

        # Normalize todos. Be tolerant of what weaker/local models emit: many
        # omit "id" (auto-assign by position) and sometimes drop/garble
        # "status" (default to "pending"). Only "content" is truly required.
        valid_statuses = ["pending", "in_progress", "completed"]
        normalized: List[Dict[str, str]] = []
        for index, todo in enumerate(todos, start=1):
            if not isinstance(todo, dict):
                return {
                    "formatted": "Error: Each todo must be an object. Re call the tool with correct format (mandatory).",
                    "isError": True,
                }

            content = todo.get("content")
            if not content:
                return {
                    "formatted": "Error: Todo missing required field 'content'. Re call the tool with correct format (mandatory).",
                    "isError": True,
                }

            todo_id = str(todo.get("id") or index)
            status = todo.get("status")
            if status not in valid_statuses:
                status = "pending"

            normalized.append({"id": todo_id, "content": str(content), "status": status})

        todos = normalized
        # Store the normalized todos structure in memory
        _current_plan = todos

        # Emit plan update event if session is available
        if self.session:
            await self.session.send_event(
                Event(
                    event_type="plan_update",
                    data={"plan": todos},
                )
            )

        # Format only for display using terminal_display utility
        formatted_output = format_plan_tool_output(todos)

        return {
            "formatted": formatted_output,
            "totalResults": len(todos),
            "isError": False,
        }


def get_current_plan() -> List[Dict[str, str]]:
    """Get the current plan (raw structure)."""
    return _current_plan


# Tool specification
PLAN_TOOL_SPEC = {
    "name": "plan_tool",
    "description": (
        "Track progress on multi-step tasks with a todo list (pending/in_progress/completed).\n\n"
        "Use for tasks with 3+ steps. Each call replaces the entire plan (send full list).\n\n"
        "Rules: exactly ONE task in_progress at a time. Mark completed immediately after finishing. "
        "Only mark completed when the task fully succeeded — keep in_progress if there are errors. "
        "Update frequently so the user sees progress."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of todo items",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for the todo",
                        },
                        "content": {
                            "type": "string",
                            "description": "Description of the todo task",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Current status of the todo",
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
}


async def plan_tool_handler(
    arguments: Dict[str, Any], session: Any = None
) -> tuple[str, bool]:
    tool = PlanTool(session=session)
    result = await tool.execute(arguments)
    return result["formatted"], not result.get("isError", False)
