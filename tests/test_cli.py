from contextlib import nullcontext

import pytest

from mycode.cli import _context_budget_from_config
from mycode.cli import build_agent_runner
from mycode.cli import build_chat_session
from mycode.cli import main
from mycode.cli import run_agent_command
from mycode.cli import run_agent_loop
from mycode.cli import run_chat_loop
from mycode.cli_presenter import summarize_tool_arguments
from mycode.agent import AgentEvent, AgentProgressSnapshot, AgentToolCall
from mycode.context_budget import ContextBudget
from mycode.config import LLMConfig
from mycode.conversation import Conversation
from mycode.llm import FakeLLMClient
from mycode.instructions import InstructionBundle, InstructionSource
from mycode.messages import Message
from mycode.run_outcome import AgentRunOutcome
from mycode.session import ChatSession
from mycode.session_runtime import SessionStartRequest, start_project_session
from mycode.session_store import ProjectIdentity, SessionStore
from mycode.subagents.delegate import DelegateTaskTool
from mycode.subagents.delegation import DelegationToolBatchHandler
from mycode.subagents.limits import (
    DEFAULT_MAX_CONCURRENT_DELEGATIONS,
    DEFAULT_MAX_DELEGATIONS_PER_PARENT_RUN,
)
from mycode.subagents.observability import CompositeSubAgentObserver
from mycode.tools import ToolResult


def context_budget(max_input_tokens: int) -> ContextBudget:
    return ContextBudget(
        context_window_tokens=max_input_tokens,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )


class FailingChatLLMClient:
    def __init__(self, *, stream_chunks: list[str]) -> None:
        self.stream_chunks = stream_chunks
        self.last_token_usage = None
        self.last_reasoning_char_count = 0

    def stream_complete(self, conversation: Conversation):
        yield from self.stream_chunks
        raise RuntimeError("Request timed out.")


def configured_llm() -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )


def test_context_budget_from_config_uses_token_window_and_reserves() -> None:
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
        context_window_tokens=200000,
        reserved_output_tokens=16000,
        context_safety_margin_tokens=8000,
    )

    budget = _context_budget_from_config(config)

    assert budget.context_window_tokens == 200000
    assert budget.reserved_output_tokens == 16000
    assert budget.safety_margin_tokens == 8000
    assert budget.max_input_tokens == 176000


def test_build_chat_session_routes_main_and_compact_models() -> None:
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="main-model",
        compact_model="compact-model",
        thinking_enabled=True,
        reasoning_effort="max",
    )

    session = build_chat_session(config)

    assert session.llm_client.model == "main-model"
    assert session.llm_client.thinking_enabled is True
    assert session.llm_client.reasoning_effort == "max"
    assert session.compactor is not None
    assert session.compactor.llm_client.model == "compact-model"
    assert session.compactor.llm_client.thinking_enabled is False


def test_main_prints_greeting(capsys) -> None:
    main([])

    captured = capsys.readouterr()

    assert captured.out == "mycode-project 已就绪。\n"


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_main_prints_top_level_help(help_flag, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main([help_flag])

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "轻量级 coding agent" in captured.out
    assert "agent" in captured.out
    assert "chat" in captured.out
    assert "mycode <子命令> --help" in captured.out
    assert "mycode agent --help" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("command", ["agent", "chat"])
def test_main_prints_subcommand_help_without_starting_runtime(
    command,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "mycode.cli.build_chat_session",
        lambda: pytest.fail("help must not build a chat session"),
    )
    monkeypatch.setattr(
        "mycode.cli.run_agent_command",
        lambda **kwargs: pytest.fail("help must not start the agent"),
    )

    with pytest.raises(SystemExit) as error:
        main([command, "--help"])

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert f"usage: mycode {command}" in captured.out
    assert captured.err == ""


def test_main_agent_help_explains_default_session_menu(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["agent", "--help"])

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "默认行为：" in captured.out
    assert "存在历史会话，则显示交互式会话菜单" in captured.out
    assert "没有历史会话，则自动创建新会话" in captured.out
    assert "会话选项:" in captured.out
    assert "--new" in captured.out
    assert "跳过菜单，直接创建新会话" in captured.out
    assert "续接最近使用的会话；没有历史会话时新建" in captured.out
    assert "续接指定的未删除会话" in captured.out
    assert "输出选项:" in captured.out
    assert "mycode agent --resume SESSION_ID" in captured.out
    assert captured.err == ""


def test_main_agent_uses_agent_command(monkeypatch) -> None:
    calls: list[tuple[SessionStartRequest, str]] = []

    monkeypatch.setattr(
        "mycode.cli.run_agent_command",
        lambda *, session_request, display_mode: calls.append(
            (session_request, display_mode)
        ),
    )

    main(["agent", "--verbose"])

    assert calls == [(SessionStartRequest(mode="select"), "verbose")]


@pytest.mark.parametrize(
    ("command", "target", "expected"),
    [
        (
            ["chat"],
            "mycode.cli.build_chat_session",
            "错误> Chat 启动失败：configuration unavailable\n",
        ),
        (
            ["agent"],
            "mycode.cli.run_agent_command",
            "错误> Agent 启动失败：configuration unavailable\n",
        ),
    ],
)
def test_main_reports_startup_failure_without_traceback(
    command,
    target,
    expected,
    monkeypatch,
    capsys,
) -> None:
    def fail(**kwargs):
        raise RuntimeError("configuration unavailable\nprivate detail")

    monkeypatch.setattr(target, fail)

    main(command)

    captured = capsys.readouterr()
    assert captured.out == expected
    assert captured.err == ""
    assert "Traceback" not in captured.out


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["agent"], (SessionStartRequest(mode="select"), "normal")),
        (
            ["agent", "--debug", "--resume", "session-id"],
            (
                SessionStartRequest(mode="resume", session_id="session-id"),
                "debug",
            ),
        ),
        (
            ["agent", "--continue", "--verbose"],
            (
                SessionStartRequest(mode="continue"),
                "verbose",
            ),
        ),
        (["agent", "--new"], (SessionStartRequest(mode="new"), "normal")),
    ],
)
def test_main_parses_agent_command_options(
    args: list[str],
    expected: tuple[SessionStartRequest, str],
    monkeypatch,
) -> None:
    calls: list[tuple[SessionStartRequest, str]] = []
    monkeypatch.setattr(
        "mycode.cli.run_agent_command",
        lambda *, session_request, display_mode: calls.append(
            (session_request, display_mode)
        ),
    )

    main(args)

    assert calls == [expected]


@pytest.mark.parametrize(
    "args",
    [
        ["agent", "--verbose", "--debug"],
        ["agent", "--new", "--continue"],
        ["agent", "--resume"],
    ],
)
def test_main_rejects_invalid_agent_options(args, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(args)

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "usage: mycode agent" in captured.err


def test_tool_argument_summary_hides_search_and_unknown_values() -> None:
    grep_summary = summarize_tool_arguments(
        "grep",
        {
            "query": "PRIVATE SEARCH TEXT",
            "path_pattern": "src/**/*.py",
        },
    )
    unknown_summary = summarize_tool_arguments(
        "future_tool",
        {"private_argument": "PRIVATE UNKNOWN VALUE"},
    )

    assert grep_summary == "query_chars=19, path_pattern='src/**/*.py'"
    assert "PRIVATE SEARCH TEXT" not in grep_summary
    assert unknown_summary == "argument_keys=['private_argument'], argument_count=1"
    assert "PRIVATE UNKNOWN VALUE" not in unknown_summary


def test_run_chat_loop_sends_multiple_messages_to_same_session() -> None:
    session = ChatSession(
        llm_client=FakeLLMClient(
            responses=["first reply", "second reply"],
            stream_chunk_size=6,
        )
    )
    inputs = iter(["hello", "again", "/exit"])
    outputs: list[str] = []

    run_chat_loop(
        session=session,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs[0] == "输入 /exit 或 /quit 退出。"
    context_outputs = [output for output in outputs if output.startswith("context> ")]
    assert len(context_outputs) == 2
    assert all("/ 115,712 tokens" in output for output in context_outputs)
    assert all("estimate" in output for output in context_outputs)
    assert [output for output in outputs if not output.startswith("context> ")] == [
        "输入 /exit 或 /quit 退出。",
        "assistant> ",
        "first ",
        "reply",
        "",
        "assistant> ",
        "second",
        " reply",
        "",
    ]
    assert session.conversation.get_messages() == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="first reply"),
        Message(role="user", content="again"),
        Message(role="assistant", content="second reply"),
    ]


def test_run_chat_loop_skips_empty_input() -> None:
    session = ChatSession(llm_client=FakeLLMClient(responses=["reply"]))
    inputs = iter(["", "hello", "/exit"])
    outputs: list[str] = []

    run_chat_loop(
        session=session,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs[0] == "输入 /exit 或 /quit 退出。"
    assert outputs[1].startswith("context> ")
    assert "/ 115,712 tokens" in outputs[1]
    assert outputs[2:] == [
        "assistant> ",
        "reply",
        "",
    ]


def test_run_chat_loop_exits_without_calling_llm() -> None:
    session = ChatSession(llm_client=FakeLLMClient(responses=[]))
    inputs = iter(["/quit"])
    outputs: list[str] = []

    run_chat_loop(
        session=session,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs == ["输入 /exit 或 /quit 退出。"]
    assert session.conversation.get_messages() == []


def test_run_chat_loop_outputs_context_notice_before_assistant_text() -> None:
    session = ChatSession(
        llm_client=FakeLLMClient(responses=["reply"]),
        conversation=Conversation.from_messages(
            [
                Message(role="user", content="old request " * 20),
                Message(role="assistant", content="old reply " * 20),
            ]
        ),
        context_budget=context_budget(100),
    )
    inputs = iter(["current", "/exit"])
    outputs: list[str] = []

    run_chat_loop(
        session=session,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs[0] == "输入 /exit 或 /quit 退出。"
    assert outputs[1].startswith("context> ~20 / 100 tokens (20.0% used")
    assert "messages=1/3" in outputs[1]
    assert "trimmed=2" in outputs[1]
    assert outputs[2:] == ["assistant> ", "reply", ""]


@pytest.mark.parametrize("stream_chunks", [[], ["partial reply"]])
def test_run_chat_loop_reports_stream_failure_and_continues(
    stream_chunks: list[str],
) -> None:
    session = ChatSession(
        llm_client=FailingChatLLMClient(stream_chunks=stream_chunks)
    )
    inputs = iter(["hello", "/exit"])
    outputs: list[str] = []

    run_chat_loop(
        session=session,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert [output for output in outputs if not output.startswith("context> ")] == [
        "输入 /exit 或 /quit 退出。",
        "assistant> ",
        *stream_chunks,
        "",
        "错误> 模型流式请求失败：Request timed out.",
    ]
    assert session.conversation.get_messages() == [
        Message(role="user", content="hello")
    ]


def test_run_chat_loop_reports_request_preparation_failure_and_continues() -> None:
    class FailingChatSession:
        last_model_context = None
        last_token_usage = None

        def stream_user_message(self, content: str):
            raise RuntimeError("compact request failed\ninternal detail")

    inputs = iter(["hello", "/exit"])
    outputs: list[str] = []

    run_chat_loop(
        session=FailingChatSession(),
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        "错误> 模型请求准备失败：compact request failed",
    ]


def test_run_chat_loop_reports_keyboard_interrupt_without_traceback() -> None:
    outputs: list[str] = []

    run_chat_loop(
        session=ChatSession(llm_client=FakeLLMClient(responses=[])),
        input_func=lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt()),
        output_func=outputs.append,
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        "",
        "提示> Chat 已中断。",
    ]


def test_run_chat_loop_reports_context_overflow_without_calling_model() -> None:
    client = FakeLLMClient(responses=["unused"])
    session = ChatSession(
        llm_client=client,
        context_budget=context_budget(60),
    )
    inputs = iter(["中" * 100, "/exit"])
    outputs: list[str] = []

    run_chat_loop(
        session=session,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs[0] == "输入 /exit 或 /quit 退出。"
    assert outputs[1].startswith("context> ~160 / 60 tokens (266.7% used")
    assert "messages=1/1" in outputs[1]
    assert "over_budget=True" in outputs[1]
    assert outputs[2].startswith("error> Model context exceeds")
    assert client.responses == ["unused"]


def test_run_agent_loop_streams_text_and_events() -> None:
    tool_call = AgentToolCall(
        id="call_123",
        name="read_file",
        arguments={"path": "README.md"},
    )
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="tool_call", tool_call=tool_call),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(
                        content="1 | hello",
                        metadata={"path": "README.md"},
                    ),
                ),
                AgentEvent(
                    type="artifact_warning",
                    content=(
                        "phase=tool_result, reason=permission_denied, "
                        "tool=read_file, failures=1; tool body was not persisted"
                    ),
                ),
                AgentEvent(type="text_delta", content="final"),
                AgentEvent(type="text_delta", content=" answer"),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["inspect", "/exit"])
    outputs: list[str] = []

    outcome = run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert runner.seen_messages == ["inspect"]
    assert outcome == AgentRunOutcome.from_stop_reason("final_answer")
    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        "活动> 正在读取文件：README.md",
        (
            "警告> 工具结果归档异常：phase=tool_result, "
            "reason=permission_denied, tool=read_file, failures=1; "
            "tool body was not persisted"
        ),
        "assistant> ",
        "final",
        " answer",
        "",
    ]


def test_run_agent_loop_groups_reads_and_reports_error_recovery() -> None:
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="model_start"),
                AgentEvent(type="text_delta", content="  "),
                AgentEvent(
                    type="tool_call",
                    tool_call=AgentToolCall(
                        id="missing",
                        name="read_file",
                        arguments={"path": "missing.py"},
                    ),
                ),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.failure(
                        error="File not found: missing.py"
                    ),
                ),
                AgentEvent(
                    type="tool_call",
                    tool_call=AgentToolCall(
                        id="first",
                        name="read_file",
                        arguments={"path": "main.py"},
                    ),
                ),
                AgentEvent(
                    type="tool_call",
                    tool_call=AgentToolCall(
                        id="second",
                        name="read_file",
                        arguments={"path": "config.py"},
                    ),
                ),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(content="main"),
                ),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(content="config"),
                ),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    outputs: list[str] = []
    inputs = iter(["inspect", "/exit"])

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        "活动> 正在读取文件：missing.py",
        "警告> 读取文件失败：未找到文件：missing.py",
        "活动> 上一步失败，正在尝试其他方式",
        "活动> 正在读取 2 个文件：main.py, config.py",
    ]
    assert "assistant> " not in outputs


def test_run_agent_loop_summarizes_mixed_read_only_batch() -> None:
    calls = [
        AgentToolCall(
            id="read",
            name="read_file",
            arguments={"path": "main.py"},
        ),
        AgentToolCall(
            id="search",
            name="grep",
            arguments={"query": "main", "path_pattern": "src/**/*.py"},
        ),
        AgentToolCall(
            id="artifact",
            name="read_artifact",
            arguments={"artifact_path": "C:/private/artifact.txt"},
        ),
    ]
    runner = FakeRunner(
        event_batches=[
            [
                *(AgentEvent(type="tool_call", tool_call=call) for call in calls),
                *(
                    AgentEvent(
                        type="tool_result",
                        tool_result=ToolResult.success(content="result"),
                    )
                    for _ in calls
                ),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    outputs: list[str] = []
    inputs = iter(["inspect", "/exit"])

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        (
            "活动> 正在进行只读分析：读取 1 个文件、搜索代码 1 次、"
            "读取 1 份已保存的工具结果"
        ),
    ]
    assert not any("C:/private" in output for output in outputs)


def test_run_agent_loop_distinguishes_continued_reads() -> None:
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(
                    type="tool_call",
                    tool_call=AgentToolCall(
                        id="file",
                        name="read_file",
                        arguments={"path": "main.py", "start_line": 151},
                    ),
                ),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(content="file result"),
                ),
                AgentEvent(
                    type="tool_call",
                    tool_call=AgentToolCall(
                        id="artifact",
                        name="read_artifact",
                        arguments={
                            "artifact_path": "C:/private/artifact.txt",
                            "offset_chars": 3000,
                        },
                    ),
                ),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(content="artifact result"),
                ),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    outputs: list[str] = []
    inputs = iter(["inspect", "/exit"])

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        "活动> 正在继续读取 main.py（从第 151 行开始）",
        "活动> 正在继续读取已保存的工具结果（从第 3000 个字符开始）",
    ]
    assert not any("C:/private" in output for output in outputs)


def test_run_agent_loop_displays_instruction_sources_and_warnings() -> None:
    runner = FakeRunner(event_batches=[])
    runner.instruction_sources = ("project: C:/workspace/AGENTS.md",)
    runner.instruction_warnings = (
        "file_too_large: C:/workspace/src/AGENTS.md: too large",
    )
    inputs = iter(["/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert outputs == [
        "instructions> 已加载 project: C:/workspace/AGENTS.md",
        (
            "instructions> 警告：file_too_large: "
            "C:/workspace/src/AGENTS.md: too large"
        ),
        "输入 /exit 或 /quit 退出。",
    ]


def test_build_agent_runner_injects_loaded_instructions(
    tmp_path, monkeypatch
) -> None:
    instruction_file = tmp_path / "AGENTS.md"
    instruction_file.write_text("project rules", encoding="utf-8")
    bundle = InstructionBundle(
        sources=(
            InstructionSource(
                path=instruction_file.resolve(),
                scope="project",
                content="project rules",
            ),
        )
    )
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="main-model",
        compact_model="compact-model",
        subagent_model="subagent-model",
        memory_context_tokens=321,
        thinking_enabled=True,
        reasoning_effort="max",
    )
    monkeypatch.setattr("mycode.cli.load_instruction_bundle", lambda root: bundle)

    runner = build_agent_runner(workspace_path=tmp_path, llm_config=config)

    system_message = runner.conversation.get_messages()[0]
    assert system_message.role == "system"
    assert "project rules" in system_message.content
    assert runner.instruction_sources == (
        f"project: {instruction_file.resolve()}",
    )
    schema_names = [
        schema["name"] for schema in runner.tool_registry.get_schemas()
    ]
    memory_start = schema_names.index("list_memories")
    assert schema_names[memory_start : memory_start + 4] == [
        "list_memories",
        "save_memory",
        "delete_memory",
        "delegate_task",
    ]
    assert schema_names[-3:] == [
        "load_skill",
        "read_skill_resource",
        "run_skill_script",
    ]
    assert runner.memory_context_selector is not None
    assert runner.memory_context_selector.policy.max_tokens == 321
    assert "长期记忆边界" in system_message.content
    assert "SubAgent 委派边界" in system_message.content
    assert isinstance(runner.tool_batch_handler, DelegationToolBatchHandler)
    assert runner.tool_batch_handler.max_concurrent_delegations == (
        DEFAULT_MAX_CONCURRENT_DELEGATIONS
    )
    assert runner.tool_batch_handler.max_delegations_per_run == (
        DEFAULT_MAX_DELEGATIONS_PER_PARENT_RUN
    )
    delegate_tool = runner.tool_registry.require("delegate_task")
    assert isinstance(delegate_tool, DelegateTaskTool)
    assert delegate_tool.runtime.workspace.root == tmp_path.resolve()
    assert delegate_tool.runtime.memory_store is not None
    assert delegate_tool.runtime.context_budget == runner.context_budget
    assert runner.llm_client.model == "main-model"
    assert runner.llm_client.thinking_enabled is True
    assert runner.llm_client.reasoning_effort == "max"
    assert runner.compactor is not None
    assert runner.compactor.llm_client.model == "compact-model"
    assert runner.compactor.llm_client.thinking_enabled is False
    subagent_client = delegate_tool.runtime.llm_client_factory()
    assert subagent_client.model == "subagent-model"
    assert subagent_client.thinking_enabled is True
    assert subagent_client.reasoning_effort == "max"


def test_build_agent_runner_registers_session_scoped_artifact_reader(
    tmp_path, monkeypatch
) -> None:
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )
    monkeypatch.setattr("mycode.cli.load_llm_config", lambda: config)
    artifact_directory = tmp_path / "state" / "artifacts" / "session"

    with pytest.raises(ValueError, match="must be provided together"):
        build_agent_runner(
            workspace_path=tmp_path,
            artifact_directory=artifact_directory,
        )

    runner = build_agent_runner(
        workspace_path=tmp_path,
        artifact_directory=artifact_directory,
        artifact_write_guard=nullcontext,
    )

    assert runner.tool_result_artifact_store is not None
    assert runner.tool_result_artifact_store.root == artifact_directory.resolve()
    assert runner.tool_result_artifact_store.write_guard is nullcontext
    assert runner.tool_registry.require("read_artifact").name == "read_artifact"


def test_build_agent_runner_restores_history_under_current_system_prompt(
    tmp_path, monkeypatch
) -> None:
    user_directory = tmp_path / "user-instructions"
    workspace = tmp_path / "workspace"
    user_directory.mkdir()
    workspace.mkdir()
    instruction_file = workspace / "MYCODE.md"
    instruction_file.write_text("current mycode rules", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("ignored agents rules", encoding="utf-8")
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
    )
    history = Conversation.from_messages(
        [
            Message(role="user", content="old request"),
            Message(role="assistant", content="old reply"),
        ]
    )
    observed: list[Message] = []
    monkeypatch.setattr(
        "mycode.instructions.default_user_instruction_directory",
        lambda: user_directory,
    )
    monkeypatch.setattr("mycode.cli.load_llm_config", lambda: config)

    runner = build_agent_runner(
        workspace_path=workspace,
        conversation_history=history,
        on_message_added=observed.append,
    )

    messages = runner.conversation.get_messages()
    assert [message.role for message in messages] == ["system", "user", "assistant"]
    assert "current mycode rules" in messages[0].content
    assert "ignored agents rules" not in messages[0].content
    assert "old request" not in messages[0].content
    assert runner.instruction_sources == (
        f"project: {instruction_file.resolve()}",
    )
    runner.conversation.add_user_message("new request")
    assert observed == [Message(role="user", content="new request")]


def test_run_agent_loop_outputs_context_notice_without_history_content() -> None:
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(
                    type="context",
                    content=(
                        "~11 / 40 tokens (27.5% used, 72.5% left, "
                        "conservative estimate, calibration pending), "
                        "messages=1/3, trimmed=2, compressed=0, "
                        "over_budget=False"
                    ),
                ),
                AgentEvent(type="model_start"),
                AgentEvent(type="text_delta", content="final answer"),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["inspect", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
        display_mode="verbose",
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        (
            "context> ~11 / 40 tokens (27.5% used, 72.5% left, "
            "conservative estimate, calibration pending), messages=1/3, "
            "trimmed=2, compressed=0, over_budget=False"
        ),
        "assistant> ",
        "final answer",
        "",
    ]
    assert not any("old request" in output or "old reply" in output for output in outputs)


def test_run_agent_loop_debug_outputs_final_stop() -> None:
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="context", content="safe context summary"),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["inspect", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
        display_mode="debug",
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        "context> safe context summary",
        "stop> final_answer",
    ]


def test_run_agent_loop_debug_outputs_safe_structured_progress() -> None:
    progress = AgentProgressSnapshot(
        stagnation_turns=2,
        reason="repetition_observed",
    )
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="progress", progress=progress),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["inspect", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
        display_mode="debug",
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        (
            "progress> stagnation_turns=2 same_tool_repeat=0 same_result_repeat=0 "
            "resource_repeat=0 convergence_guided=False "
            "reason=repetition_observed"
        ),
        "stop> final_answer",
    ]


def test_run_agent_loop_explains_max_turns_in_chinese() -> None:
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="text_delta", content="阶段性结论"),
                AgentEvent(
                    type="stop",
                    content=(
                        "本轮已达到 50 轮上限；上面是基于现有信息整理的"
                        "阶段性结果。"
                    ),
                    stop_reason="max_turns",
                ),
            ]
        ]
    )
    inputs = iter(["inspect", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        "assistant> ",
        "阶段性结论",
        "",
        "警告> 本轮已达到 50 轮上限；上面是基于现有信息整理的阶段性结果。",
        "提示> 当前进度已保存；如需继续，直接输入“继续”。",
    ]


def test_run_agent_loop_displays_turn_progress_and_convergence_notice() -> None:
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="turn", turn_number=18, max_turns=50),
                AgentEvent(
                    type="turn",
                    content="剩余约 5 轮，Agent 将优先收敛",
                    turn_number=46,
                    max_turns=50,
                ),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["inspect", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert outputs == [
        "输入 /exit 或 /quit 退出。",
        "轮次> 18/50",
        "轮次> 46/50",
        "提醒> 剩余约 5 轮，Agent 将优先收敛",
    ]


def test_run_agent_loop_summarizes_write_file_arguments() -> None:
    tool_call = AgentToolCall(
        id="call_write",
        name="write_file",
        arguments={"path": "notes.txt", "content": "secret content"},
    )
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="tool_call", tool_call=tool_call),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(
                        content="Wrote file: notes.txt",
                        metadata={"path": "notes.txt"},
                    ),
                ),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["write", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
        display_mode="verbose",
    )

    assert "tool_call> write_file(path='notes.txt', content_chars=14)" in outputs
    assert not any("secret content" in output for output in outputs)


def test_run_agent_loop_summarizes_edit_file_arguments() -> None:
    tool_call = AgentToolCall(
        id="call_edit",
        name="edit_file",
        arguments={"path": "notes.txt", "old_text": "old", "new_text": "new text"},
    )
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="tool_call", tool_call=tool_call),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(
                        content="Edited file: notes.txt",
                        metadata={"path": "notes.txt"},
                    ),
                ),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["edit", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
        display_mode="verbose",
    )

    assert (
        "tool_call> edit_file(path='notes.txt', old_text_chars=3, new_text_chars=8)"
        in outputs
    )
    assert not any("'old'" in output or "new text" in output for output in outputs)


def test_run_agent_loop_summarizes_run_command_arguments() -> None:
    tool_call = AgentToolCall(
        id="call_command",
        name="run_command",
        arguments={
            "command": ["uv", "run", "pytest"],
            "cwd": ".",
            "timeout_seconds": 30,
        },
    )
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="tool_call", tool_call=tool_call),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(
                        content="Command exited with code 0.",
                        metadata={"exit_code": 0},
                    ),
                ),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["test", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
        display_mode="verbose",
    )

    assert (
        "tool_call> run_command(command='uv run pytest', cwd='.', timeout_seconds=30)"
        in outputs
    )


def test_run_agent_loop_summarizes_delegate_without_task_or_child_body() -> None:
    tool_call = AgentToolCall(
        id="call_delegate",
        name="delegate_task",
        arguments={
            "role": "explorer",
            "objective": "PRIVATE OBJECTIVE BODY",
            "context": "PRIVATE CONTEXT BODY",
            "scope_paths": ["private/path.py"],
        },
    )
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="tool_call", tool_call=tool_call),
                AgentEvent(
                    type="tool_result",
                    tool_result=ToolResult.success(
                        content='{"summary":"PRIVATE CHILD RESULT BODY"}',
                        metadata={
                            "tool_name": "delegate_task",
                            "role": "explorer",
                            "child_status": "completed",
                            "child_stop_reason": "submitted",
                        },
                    ),
                ),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    outputs: list[str] = []
    inputs = iter(["delegate", "/exit"])

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
        display_mode="verbose",
    )

    assert (
        "tool_call> delegate_task(role='explorer', objective_chars=22, "
        "context_chars=20, scope_path_count=1)"
    ) in outputs
    assert "tool_result> ok: subagent role=explorer status=completed stop=submitted" in outputs
    assert not any(
        secret in output
        for output in outputs
        for secret in (
            "PRIVATE OBJECTIVE BODY",
            "PRIVATE CONTEXT BODY",
            "private/path.py",
            "PRIVATE CHILD RESULT BODY",
        )
    )


def test_run_agent_loop_hides_save_memory_content_before_confirmation() -> None:
    tool_call = AgentToolCall(
        id="call-memory",
        name="save_memory",
        arguments={
            "scope": "user",
            "kind": "preference",
            "key": "response.style",
            "content": "private preference text",
        },
    )
    runner = FakeRunner(
        event_batches=[
            [
                AgentEvent(type="tool_call", tool_call=tool_call),
                AgentEvent(type="stop", stop_reason="final_answer"),
            ]
        ]
    )
    inputs = iter(["remember", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
        display_mode="verbose",
    )

    assert (
        "tool_call> save_memory(scope='user', kind='preference', "
        "key='response.style', content_chars=23)"
    ) in outputs
    assert not any("private preference text" in output for output in outputs)


def test_run_agent_loop_skips_empty_input() -> None:
    runner = FakeRunner(event_batches=[[AgentEvent(type="stop", stop_reason="final_answer")]])
    inputs = iter(["", "/exit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert runner.seen_messages == []
    assert outputs == ["输入 /exit 或 /quit 退出。"]


def test_run_agent_loop_exits_without_calling_runner() -> None:
    runner = FakeRunner(event_batches=[])
    inputs = iter(["/quit"])
    outputs: list[str] = []

    run_agent_loop(
        runner=runner,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_chunk_func=outputs.append,
    )

    assert runner.seen_messages == []
    assert outputs == ["输入 /exit 或 /quit 退出。"]


def test_run_agent_command_uses_same_io_for_confirmer(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    inputs = iter(["/quit"])
    outputs: list[str] = []

    def fake_input(prompt: str) -> str:
        return next(inputs)

    def fake_output(message: str) -> None:
        outputs.append(message)

    def fake_build_agent_runner(
        *,
        workspace_path=None,
        confirmer=None,
        conversation_history=None,
        on_message_added=None,
        compact_state=None,
        on_compact_state_changed=None,
        artifact_directory=None,
        artifact_write_guard=None,
        subagent_observer=None,
        llm_config=None,
    ):
        captured["workspace_path"] = workspace_path
        captured["confirmer"] = confirmer
        captured["conversation_history"] = conversation_history
        captured["on_message_added"] = on_message_added
        captured["compact_state"] = compact_state
        captured["on_compact_state_changed"] = on_compact_state_changed
        captured["artifact_directory"] = artifact_directory
        captured["artifact_write_guard"] = artifact_write_guard
        captured["subagent_observer"] = subagent_observer
        captured["llm_config"] = llm_config
        return FakeRunner(event_batches=[])

    monkeypatch.setattr("mycode.cli.build_agent_runner", fake_build_agent_runner)

    run_agent_command(
        workspace_path=tmp_path,
        input_func=fake_input,
        output_func=fake_output,
        display_mode="debug",
        session_request=SessionStartRequest(mode="new"),
        session_store=SessionStore(tmp_path / "state.sqlite3"),
        llm_config=configured_llm(),
    )

    confirmer = captured["confirmer"]
    assert confirmer.input_func is fake_input
    assert confirmer.output_func is fake_output
    assert captured["workspace_path"] == tmp_path.resolve()
    assert captured["conversation_history"].get_messages() == []
    assert callable(captured["on_message_added"])
    assert captured["compact_state"].boundary is None
    assert callable(captured["on_compact_state_changed"])
    assert captured["artifact_directory"].is_absolute()
    assert callable(captured["artifact_write_guard"])
    assert isinstance(captured["subagent_observer"], CompositeSubAgentObserver)
    cli_observer = captured["subagent_observer"].observers[1]
    assert cli_observer.mode == "debug"
    assert outputs[0].startswith("session> 已创建新会话 ")
    assert outputs[1:] == ["输入 /exit 或 /quit 退出。"]


def test_run_agent_command_persists_runner_messages_and_closes_session(
    tmp_path, monkeypatch
) -> None:
    store = SessionStore(tmp_path / "state.sqlite3")

    class PersistingFakeRunner:
        instruction_sources = ()
        instruction_warnings = ()

        def __init__(self, on_message_added) -> None:
            self.on_message_added = on_message_added

        def run(self, user_message: str):
            self.on_message_added(Message(role="user", content=user_message))
            self.on_message_added(Message(role="assistant", content="persisted reply"))
            yield AgentEvent(type="stop", stop_reason="final_answer")

    def fake_build_agent_runner(**kwargs):
        return PersistingFakeRunner(kwargs["on_message_added"])

    monkeypatch.setattr("mycode.cli.build_agent_runner", fake_build_agent_runner)
    inputs = iter(["persist this", "/exit"])
    outputs: list[str] = []

    run_agent_command(
        workspace_path=tmp_path,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        session_request=SessionStartRequest(mode="new"),
        session_store=store,
        llm_config=configured_llm(),
    )

    project = ProjectIdentity.from_workspace(tmp_path)
    sessions = store.list_sessions(project)
    assert len(sessions) == 1
    assert sessions[0].title == "persist this"
    assert sessions[0].status == "closed"
    assert store.load_conversation(project, sessions[0].id).get_messages() == [
        Message(role="user", content="persist this"),
        Message(role="assistant", content="persisted reply"),
    ]


def test_run_agent_command_marks_session_interrupted_on_unexpected_error(
    tmp_path, monkeypatch
) -> None:
    store = SessionStore(tmp_path / "state.sqlite3")

    class FailingRunner:
        instruction_sources = ()
        instruction_warnings = ()

        def run(self, user_message: str):
            raise RuntimeError("unexpected failure")
            yield

    monkeypatch.setattr("mycode.cli.build_agent_runner", lambda **kwargs: FailingRunner())
    inputs = iter(["trigger failure"])
    outputs: list[str] = []

    run_agent_command(
        workspace_path=tmp_path,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        session_request=SessionStartRequest(mode="new"),
        session_store=store,
        llm_config=configured_llm(),
    )

    project = ProjectIdentity.from_workspace(tmp_path)
    sessions = store.list_sessions(project)
    assert len(sessions) == 1
    assert sessions[0].status == "interrupted"
    assert outputs[-1] == "错误> Agent 运行失败：unexpected failure"
    assert all("Traceback" not in output for output in outputs)


def test_run_agent_command_marks_session_interrupted_on_keyboard_interrupt(
    tmp_path, monkeypatch
) -> None:
    store = SessionStore(tmp_path / "state.sqlite3")

    class InterruptedRunner:
        instruction_sources = ()
        instruction_warnings = ()

        def run(self, user_message: str):
            raise KeyboardInterrupt
            yield

    monkeypatch.setattr(
        "mycode.cli.build_agent_runner", lambda **kwargs: InterruptedRunner()
    )
    outputs: list[str] = []

    run_agent_command(
        workspace_path=tmp_path,
        input_func=lambda prompt: "interrupt me",
        output_func=outputs.append,
        session_request=SessionStartRequest(mode="new"),
        session_store=store,
        llm_config=configured_llm(),
    )

    project = ProjectIdentity.from_workspace(tmp_path)
    sessions = store.list_sessions(project)
    assert len(sessions) == 1
    assert sessions[0].status == "interrupted"
    assert outputs[-2:] == [
        "",
        "提示> Agent 已中断，当前进度已保存。",
    ]


def test_run_agent_command_reports_session_owned_by_another_agent(tmp_path) -> None:
    store = SessionStore(tmp_path / "state.sqlite3")
    project = ProjectIdentity.from_workspace(tmp_path)
    first = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
        output_func=lambda message: None,
    )
    assert first is not None
    outputs: list[str] = []

    run_agent_command(
        workspace_path=tmp_path,
        input_func=lambda prompt: pytest.fail("agent loop must not start"),
        output_func=outputs.append,
        session_request=SessionStartRequest(
            mode="resume",
            session_id=first.record.id,
        ),
        session_store=SessionStore(store.database_path),
        llm_config=configured_llm(),
    )

    assert outputs == [
        f"session> 错误：Session is already in use by another agent: "
        f"{first.record.id}"
    ]
    assert store.get_session(project, first.record.id).status == "active"
    first.close()


class FakeRunner:
    def __init__(self, event_batches: list[list[AgentEvent]]) -> None:
        self.event_batches = event_batches
        self.seen_messages: list[str] = []

    def run(self, user_message: str):
        self.seen_messages.append(user_message)

        if not self.event_batches:
            raise RuntimeError("FakeRunner has no event batches left")

        yield from self.event_batches.pop(0)
