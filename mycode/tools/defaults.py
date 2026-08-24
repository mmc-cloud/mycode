from collections.abc import Iterable

from mycode.permissions import Confirmer
from mycode.memory import MemoryStore
from mycode.tools.base import BaseTool
from mycode.tools.edit_file import EditFileTool
from mycode.tools.glob import GlobTool
from mycode.tools.grep import GrepTool
from mycode.tools.read_file import ReadFileTool
from mycode.tools.registry import ToolRegistry
from mycode.tools.run_command import RunCommandTool
from mycode.tools.run_validation import RunValidationTool
from mycode.tools.memory import DeleteMemoryTool, ListMemoriesTool, SaveMemoryTool
from mycode.tools.workspace import Workspace
from mycode.tools.write_file import WriteFileTool


def create_read_only_tool_registry(
    workspace: Workspace,
    *,
    confirmer: Confirmer | None = None,
) -> ToolRegistry:
    return ToolRegistry.from_tools(
        [
            ReadFileTool(workspace),
            GlobTool(workspace),
            GrepTool(workspace),
        ],
        confirmer=confirmer,
    )


def create_default_tool_registry(
    workspace: Workspace,
    *,
    confirmer: Confirmer | None = None,
    memory_store: MemoryStore | None = None,
    extra_tools: Iterable[BaseTool] = (),
) -> ToolRegistry:
    tools = [
        ReadFileTool(workspace),
        GlobTool(workspace),
        GrepTool(workspace),
        WriteFileTool(workspace),
        EditFileTool(workspace),
        RunCommandTool(workspace),
        RunValidationTool(workspace, restrict_to_known_validators=False),
    ]
    if memory_store is not None:
        tools.extend(
            [
                ListMemoriesTool(memory_store),
                SaveMemoryTool(memory_store),
                DeleteMemoryTool(memory_store),
            ]
        )
    tools.extend(extra_tools)
    return ToolRegistry.from_tools(
        tools,
        confirmer=confirmer,
    )
