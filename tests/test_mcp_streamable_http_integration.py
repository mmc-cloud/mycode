import socket
import subprocess
import sys
import time
from pathlib import Path

from mycode.agent import AgentToolCall
from mycode.mcp.config import MCPConfig
from mycode.mcp.manager import MCPManager
from mycode.runner import execute_tool_batch
from mycode.tools import ToolRegistry


def test_local_streamable_http_headers_calls_timeout_and_cleanup() -> None:
    port = _free_port()
    fixture = Path(__file__).parent / "fixtures" / "mcp_test_server.py"
    process = subprocess.Popen(
        [sys.executable, str(fixture), "streamable-http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    manager = MCPManager(_http_config(port, header="secret"))
    try:
        _wait_for_port(port, process)
        manager.start()
        assert manager.statuses[0].status == "connected"
        registry = ToolRegistry.from_tools(manager.tools)
        batch = execute_tool_batch(
            registry,
            [
                AgentToolCall(
                    id="1",
                    name="mcp__local__echo",
                    arguments={"text": "hi"},
                ),
                AgentToolCall(
                    id="2",
                    name="mcp__local__add",
                    arguments={"a": 2, "b": 3},
                ),
                AgentToolCall(
                    id="3", name="mcp__local__fail", arguments={}
                ),
                AgentToolCall(
                    id="4", name="mcp__local__slow", arguments={}
                ),
            ],
        )

        assert batch.executions[0].result.content == "hi"
        assert batch.executions[1].result.metadata["structured_content"] == {
            "sum": 5
        }
        assert batch.executions[2].result.ok is False
        assert batch.executions[3].result.metadata["error_type"] == "TimeoutError"
    finally:
        manager.close()
        process.terminate()
        process.wait(timeout=5)

    assert manager.shutdown_status.status == "completed"
    assert manager._clients == {}
    assert manager._thread is None


def test_local_streamable_http_rejects_missing_configured_header() -> None:
    port = _free_port()
    fixture = Path(__file__).parent / "fixtures" / "mcp_test_server.py"
    process = subprocess.Popen(
        [sys.executable, str(fixture), "streamable-http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    manager = MCPManager(_http_config(port, header="wrong"))
    try:
        _wait_for_port(port, process)
        manager.start()
        assert manager.statuses[0].status == "failed"
        assert manager.statuses[0].error_summary == "Authentication failed (401)"
    finally:
        manager.close()
        process.terminate()
        process.wait(timeout=5)


def _http_config(port: int, *, header: str) -> MCPConfig:
    return MCPConfig.model_validate(
        {
            "mcpServers": {
                "local": {
                    "transport": "streamable_http",
                    "url": f"http://127.0.0.1:{port}/mcp",
                    "headers": {"X-MyCode-Test": header},
                    "connect_timeout": 5,
                    "tool_timeout": 0.1,
                }
            }
        }
    )


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("MCP HTTP fixture exited before startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise TimeoutError("MCP HTTP fixture did not start")
