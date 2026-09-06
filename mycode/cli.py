import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from mycode.agent import AgentEvent
from mycode.application import (
    build_agent_runner,
    context_budget_from_config,
    run_agent_turn,
)
from mycode.confirmers import TerminalConfirmer
from mycode.cli_presenter import CliDisplayMode, CliPresenter
from mycode.config import LLMConfig, load_llm_config
from mycode.context_compact import ConversationCompactor
from mycode.context_budget import (
    ContextBudgetExceededError,
    format_model_context_stats,
)
from mycode.error_handling import error_summary, format_model_error
from mycode.conversation import Conversation
from mycode.mcp import (
    MCPConfig,
    MCPConfigError,
    MCPManager,
    apply_project_mcp_trust,
    load_mcp_config_layers,
)
from mycode.observability import ObservationSink
from mycode.llm import OpenAICompatibleLLMClient
from mycode.project import ProjectIdentity
from mycode.runner import AgentRunner
from mycode.run_outcome import AgentRunOutcome
from mycode.session import ChatSession
from mycode.session_runtime import SessionStartRequest, start_project_session
from mycode.session_store import (
    SessionDatabaseCorruptionError,
    SessionInUseError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)
from mycode.subagents.cli_observer import CliSubAgentObserver
from mycode.subagents.observability import (
    CompositeSubAgentObserver,
)
from mycode.subagents.persistence import SessionSubAgentObserver
from mycode.tools import (
    Workspace,
)


EXIT_COMMANDS = {"/exit", "/quit"}


def build_chat_session(llm_config: LLMConfig | None = None) -> ChatSession:
    config = load_llm_config() if llm_config is None else llm_config
    client = OpenAICompatibleLLMClient(config=config)
    summary_client = OpenAICompatibleLLMClient(
        config=config,
        model=config.compact_model,
        thinking_enabled=False,
    )

    return ChatSession(
        llm_client=client,
        context_budget=context_budget_from_config(config),
        compactor=ConversationCompactor(llm_client=summary_client),
    )


def run_chat_loop(
    session: ChatSession,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    output_chunk_func: Callable[[str], None] | None = None,
) -> None:
    if output_chunk_func is None:
        output_chunk_func = _print_chunk

    output_func("输入 /exit 或 /quit 退出。")

    while True:
        try:
            content = input_func("you> ").strip()
        except EOFError:
            output_func("")
            break
        except KeyboardInterrupt:
            output_func("")
            output_func("提示> Chat 已中断。")
            break

        if content in EXIT_COMMANDS:
            break

        if content == "":
            continue

        try:
            chunks = iter(session.stream_user_message(content))
        except ContextBudgetExceededError as error:
            output_func(
                "context> "
                + format_model_context_stats(
                    error.context,
                    previous_prompt_tokens=(
                        session.last_token_usage.prompt_tokens
                        if session.last_token_usage is not None
                        else None
                    ),
                )
            )
            output_func(f"error> {error}")
            continue
        except KeyboardInterrupt:
            output_func("")
            output_func("提示> Chat 已中断。")
            break
        except Exception as error:
            output_func(
                "错误> "
                + format_model_error(error, operation="模型请求准备失败")
            )
            continue

        context = session.last_model_context
        if context is not None:
            output_func(
                "context> "
                + format_model_context_stats(
                    context,
                    previous_prompt_tokens=(
                        session.last_token_usage.prompt_tokens
                        if session.last_token_usage is not None
                        else None
                    ),
                )
            )

        output_chunk_func("assistant> ")
        try:
            for chunk in chunks:
                output_chunk_func(chunk)
        except KeyboardInterrupt:
            output_func("")
            output_func("提示> Chat 已中断。")
            break
        except Exception as error:
            output_func("")
            output_func(
                "错误> "
                + format_model_error(error, operation="模型流式请求失败")
            )
            continue

        output_func("")


def run_agent_loop(
    runner: AgentRunner,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    output_chunk_func: Callable[[str], None] | None = None,
    display_mode: CliDisplayMode = "normal",
) -> AgentRunOutcome | None:
    if output_chunk_func is None:
        output_chunk_func = _print_chunk

    presenter = CliPresenter(output=output_func, mode=display_mode)

    for source in getattr(runner, "instruction_sources", ()):
        output_func(f"instructions> 已加载 {source}")
    for warning in getattr(runner, "instruction_warnings", ()):
        output_func(f"instructions> 警告：{warning}")
    for warning in getattr(runner, "skill_warnings", ()):
        output_func(f"skills> 警告：{warning}")

    output_func("输入 /exit 或 /quit 退出。")

    last_outcome: AgentRunOutcome | None = None
    while True:
        try:
            content = input_func("you> ").strip()
        except EOFError:
            output_func("")
            break

        if content in EXIT_COMMANDS:
            break

        if content == "":
            continue

        last_outcome = _run_agent_turn(
            runner=runner,
            content=content,
            output_func=output_func,
            output_chunk_func=output_chunk_func,
            presenter=presenter,
        )

    return last_outcome


def run_agent_command(
    workspace_path: Path | None = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    output_chunk_func: Callable[[str], None] | None = None,
    display_mode: CliDisplayMode = "normal",
    *,
    session_request: SessionStartRequest | None = None,
    session_store: SessionStore | None = None,
    llm_config: LLMConfig | None = None,
    mcp_config: MCPConfig | None = None,
    observability_sink: ObservationSink | None = None,
) -> None:
    workspace = Workspace(Path.cwd() if workspace_path is None else workspace_path)
    project = ProjectIdentity.from_workspace(workspace.root)
    config = (
        load_llm_config(workspace_root=workspace.root)
        if llm_config is None
        else llm_config
    )
    try:
        store = SessionStore() if session_store is None else session_store
        active_session = start_project_session(
            store,
            project,
            request=session_request,
            input_func=input_func,
            output_func=output_func,
        )
    except SessionDatabaseCorruptionError as error:
        output_func(f"session> 严重错误：{error}")
        return
    except (SessionNotFoundError, SessionInUseError) as error:
        output_func(f"session> 错误：{error}")
        return
    if active_session is None:
        return

    mcp_manager = MCPManager(MCPConfig(), observability_sink=observability_sink)
    try:
        active_session.start_heartbeat()
        confirmer = TerminalConfirmer(
            input_func=input_func,
            output_func=output_func,
        )
        subagent_observer = CompositeSubAgentObserver(
            observers=(
                SessionSubAgentObserver(
                    store=store,
                    project=project,
                    parent_session_id=active_session.record.id,
                    lease_owner_id=active_session.lease_owner_id,
                ),
                CliSubAgentObserver(output=output_func, mode=display_mode),
            )
        )
        conversation_history = active_session.load_history()
        compact_state = active_session.load_compact_state()
        if active_session.compact_state_recovered:
            output_func(
                "session> 警告：无效的 Compact 状态已重置；"
                "已恢复完整历史并进入 Compact 冷却期"
            )
        try:
            if mcp_config is None:
                loaded_mcp_config = load_mcp_config_layers(
                    workspace_root=workspace.root
                )
                effective_mcp_config = apply_project_mcp_trust(
                    loaded_mcp_config,
                    project,
                    input_func=input_func,
                    output_func=output_func,
                )
            else:
                effective_mcp_config = mcp_config
        except MCPConfigError as error:
            output_func("MCP servers:")
            output_func(f"✗ config      {error}")
            effective_mcp_config = MCPConfig()
        mcp_manager = MCPManager(
            effective_mcp_config,
            observability_sink=observability_sink,
        )
        mcp_manager.start()
        if mcp_manager.statuses:
            output_func("MCP servers:")
            for status in mcp_manager.statuses:
                if status.status == "connected":
                    output_func(f"✓ {status.alias:<12} {status.tool_count} tools")
                else:
                    output_func(
                        f"✗ {status.alias:<12} "
                        f"{status.error_summary or status.error_type or 'unavailable'}"
                    )
        runner = build_agent_runner(
            workspace_path=workspace.root,
            confirmer=confirmer,
            conversation_history=conversation_history,
            on_message_added=active_session.persist_message,
            compact_state=compact_state,
            on_compact_state_changed=active_session.persist_compact_state,
            artifact_directory=active_session.artifact_directory,
            artifact_write_guard=active_session.artifact_write_guard,
            subagent_observer=subagent_observer,
            llm_config=config,
        )
        for tool in mcp_manager.tools:
            runner.tool_registry.register(tool)
        run_agent_loop(
            runner=runner,
            input_func=input_func,
            output_func=output_func,
            output_chunk_func=output_chunk_func,
            display_mode=display_mode,
        )
    except KeyboardInterrupt:
        try:
            active_session.interrupt()
        except SessionStoreError as lifecycle_error:
            output_func(f"session> 警告：{lifecycle_error}")
        output_func("")
        output_func("提示> Agent 已中断，当前进度已保存。")
    except Exception as error:
        try:
            active_session.interrupt()
        except SessionStoreError as lifecycle_error:
            output_func(f"session> 警告：{lifecycle_error}")
        output_func("")
        output_func(
            "错误> " + format_model_error(error, operation="Agent 运行失败")
        )
    else:
        active_session.close()
    finally:
        mcp_manager.close()


def _run_agent_turn(
    *,
    runner: AgentRunner,
    content: str,
    output_func: Callable[[str], None],
    output_chunk_func: Callable[[str], None],
    presenter: CliPresenter,
) -> AgentRunOutcome:
    assistant_started = False

    def show_event(event: AgentEvent) -> None:
        nonlocal assistant_started
        if event.type == "model_start":
            return

        if event.type == "text_delta":
            if not assistant_started and event.content.strip() == "":
                return
            if not assistant_started:
                presenter.flush()
                output_chunk_func("assistant> ")
                assistant_started = True
            output_chunk_func(event.content)
            return

        if assistant_started:
            output_func("")
            assistant_started = False

        presenter.show_agent_event(event)

    outcome = run_agent_turn(
        runner,
        content,
        event_handler=show_event,
    )

    if assistant_started:
        output_func("")
    presenter.flush()
    return outcome


def _print_chunk(content: str) -> None:
    print(content, end="", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    if args == []:
        print("mycode-project 已就绪。")
        return

    parser = _build_cli_parser()
    options = parser.parse_args(args)

    if options.command == "chat":
        try:
            run_chat_loop(build_chat_session())
        except KeyboardInterrupt:
            print("")
            print("提示> Chat 已中断。")
        except Exception as error:
            print(f"错误> Chat 启动失败：{error_summary(error)}")
        return

    session_request = SessionStartRequest(mode="select")
    if options.new:
        session_request = SessionStartRequest(mode="new")
    elif options.continue_session:
        session_request = SessionStartRequest(mode="continue")
    elif options.resume is not None:
        session_request = SessionStartRequest(
            mode="resume",
            session_id=options.resume,
        )

    display_mode: CliDisplayMode = "normal"
    if options.verbose:
        display_mode = "verbose"
    elif options.debug:
        display_mode = "debug"

    try:
        run_agent_command(
            session_request=session_request,
            display_mode=display_mode,
        )
    except KeyboardInterrupt:
        print("")
        print("提示> Agent 已中断。")
    except Exception as error:
        print(f"错误> Agent 启动失败：{error_summary(error)}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mycode",
        add_help=False,
        description="轻量级 coding agent，以启动命令时的当前目录作为工作区。",
        epilog=(
            "示例：\n"
            "  mycode agent\n"
            "  mycode agent --new --verbose\n"
            "  mycode agent --resume SESSION_ID\n"
            "\n"
            "使用 'mycode <子命令> --help' 查看对应帮助，"
            "例如 'mycode agent --help'。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_help_argument(parser)
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        required=True,
    )

    chat_parser = subparsers.add_parser(
        "chat",
        add_help=False,
        help="启动不带 coding tools 的普通模型对话",
        description="启动不带文件、命令等 coding tools 的普通模型对话。",
    )
    _add_help_argument(chat_parser)

    agent_parser = subparsers.add_parser(
        "agent",
        add_help=False,
        help="在当前目录启动 coding agent",
        description=(
            "在当前目录启动 coding agent。\n"
            "\n"
            "默认行为：\n"
            "  不指定会话选项时，如果当前项目存在历史会话，"
            "则显示交互式会话菜单；\n"
            "  如果没有历史会话，则自动创建新会话。"
        ),
        epilog=(
            "示例：\n"
            "  mycode agent\n"
            "  mycode agent --new\n"
            "  mycode agent --continue\n"
            "  mycode agent --resume SESSION_ID\n"
            "  mycode agent --continue --debug"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_help_argument(agent_parser)
    session_options = agent_parser.add_argument_group("会话选项")
    session_group = session_options.add_mutually_exclusive_group()
    session_group.add_argument(
        "--new",
        action="store_true",
        help="跳过菜单，直接创建新会话",
    )
    session_group.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="跳过菜单，续接最近使用的会话；没有历史会话时新建",
    )
    session_group.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="跳过菜单，续接指定的未删除会话",
    )
    display_options = agent_parser.add_argument_group("输出选项")
    display_group = display_options.add_mutually_exclusive_group()
    display_group.add_argument(
        "--verbose",
        action="store_true",
        help="显示更完整的运行信息",
    )
    display_group.add_argument(
        "--debug",
        action="store_true",
        help="显示调试级运行信息",
    )
    return parser


def _add_help_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="显示此帮助并退出",
    )
