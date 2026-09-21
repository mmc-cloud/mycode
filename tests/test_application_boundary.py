import ast
from pathlib import Path
import subprocess
import sys


APPLICATION_ROOT = Path(__file__).parents[1] / "mycode" / "application"
FRONTEND_PATHS = (
    Path(__file__).parents[1] / "mycode" / "cli.py",
    Path(__file__).parents[1] / "mycode" / "adapters" / "jsonl.py",
    Path(__file__).parents[1] / "mycode" / "presentation" / "tui" / "app.py",
)
MCP_TRUST_PATH = Path(__file__).parents[1] / "mycode" / "mcp" / "trust.py"
CORE_ROOTS = tuple(
    Path(__file__).parents[1] / "mycode" / name
    for name in (
        "agent",
        "context",
        "persistence",
        "mcp",
        "tools",
        "subagents",
        "application",
    )
)


def test_application_modules_do_not_depend_on_presentation_or_terminal_io() -> None:
    forbidden_import_prefixes = (
        "mycode.presentation",
        "mycode.adapters",
    )
    forbidden_names = {
        "CliPresenter",
        "CliSubAgentObserver",
        "TerminalConfirmer",
    }
    violations: list[str] = []

    for path in APPLICATION_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name.startswith(prefix)
                        for prefix in forbidden_import_prefixes
                    ):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if any(
                    (node.module or "").startswith(prefix)
                    for prefix in forbidden_import_prefixes
                ):
                    violations.append(f"{path}: from {node.module}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "input",
                    "print",
                }:
                    violations.append(f"{path}: {node.func.id}()")
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{path}: {node.id}")

    assert violations == []


def test_mcp_trust_is_presentation_neutral() -> None:
    tree = ast.parse(MCP_TRUST_PATH.read_text(encoding="utf-8"), filename=str(MCP_TRUST_PATH))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.startswith("mycode.presentation")
                for alias in node.names
            ):
                violations.append(f"{MCP_TRUST_PATH}: presentation import")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("mycode.presentation"):
                violations.append(f"{MCP_TRUST_PATH}: presentation import")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"input", "print"}:
                violations.append(f"{MCP_TRUST_PATH}: {node.func.id}()")

    assert violations == []


def test_core_packages_do_not_import_presentation() -> None:
    violations: list[str] = []
    for root in CORE_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("mycode.presentation"):
                            violations.append(f"{path}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").startswith("mycode.presentation"):
                        violations.append(f"{path}: from {node.module}")

    assert violations == []


def test_frontends_delegate_startup_assembly_to_application_factory() -> None:
    forbidden_calls = {
        "load_mcp_config_layers",
        "resolve_project_mcp_trust",
        "start_agent_application_session",
        "build_agent_runner",
        "start_project_session",
        "ApplicationSessionFactory",
        "AgentApplicationSession",
    }
    violations: list[str] = []
    for path in FRONTEND_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
            else:
                continue
            if called_name in forbidden_calls:
                violations.append(f"{path}: {called_name}()")
    assert violations == []


def test_frontends_do_not_import_startup_implementation_symbols() -> None:
    forbidden_modules = {
        "mycode.application": {
            "start_agent_application_session",
            "build_agent_runner",
            "start_project_session",
        },
        "mycode.application.agent_session": {"start_agent_application_session"},
        "mycode.application.runtime": {"build_agent_runner"},
        "mycode.application.sessions": {"start_project_session"},
        "mycode.mcp.config": {"load_mcp_config_layers"},
        "mycode.mcp.trust": {"resolve_project_mcp_trust"},
        "mycode.config": {"load_llm_config"},
    }
    violations: list[str] = []
    for path in FRONTEND_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                forbidden = forbidden_modules.get(node.module or (), set())
                for name in names & forbidden:
                    violations.append(f"{path}: from {node.module} import {name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        violations.append(f"{path}: import {alias.name}")
    assert violations == []


def test_startup_factory_has_no_presentation_or_protocol_dependencies() -> None:
    path = APPLICATION_ROOT / "startup.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("mycode.presentation", "mycode.adapters")):
                    violations.append(f"{path}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith(("mycode.presentation", "mycode.adapters")):
                violations.append(f"{path}: from {node.module}")
    assert violations == []


def test_core_llm_contract_import_does_not_load_openai_sdk() -> None:
    command = (
        "import sys; "
        "import mycode.llm_contracts; "
        "import mycode.agent.runner; "
        "import mycode.subagents.runtime; "
        "assert 'openai' not in sys.modules"
    )

    subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).parents[1],
        check=True,
    )


def test_model_contract_and_provider_do_not_load_context_package() -> None:
    command = (
        "import sys; "
        "import mycode.llm_contracts; "
        "import mycode.providers.openai_compatible; "
        "assert not any(name == 'mycode.context' "
        "or name.startswith('mycode.context.') for name in sys.modules)"
    )

    subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).parents[1],
        check=True,
    )


def test_model_contract_and_openai_provider_do_not_load_agent_package() -> None:
    command = (
        "import sys; "
        "import mycode.llm_contracts; "
        "import mycode.providers.openai_compatible; "
        "assert not any(name == 'mycode.agent' or name.startswith('mycode.agent.') "
        "for name in sys.modules)"
    )

    subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).parents[1],
        check=True,
    )


def test_precise_mcp_and_tool_imports_keep_implementation_modules_unloaded() -> None:
    command = (
        "import sys; "
        "import mycode.mcp.config; "
        "assert 'mcp' not in sys.modules; "
        "assert 'mycode.mcp.manager' not in sys.modules; "
        "import mycode.tools.base; "
        "forbidden = {'mycode.tools.edit_file', 'mycode.tools.run_command', "
        "'mycode.tools.defaults'}; "
        "assert forbidden.isdisjoint(sys.modules)"
    )

    subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).parents[1],
        check=True,
    )
