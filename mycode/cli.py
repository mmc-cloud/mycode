import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from mycode.agent.events import AgentEvent
from mycode.adapters.jsonl import run_jsonl_runtime
from mycode.application.events import RuntimeEvent
from mycode.application import (
    AgentApplicationSession,
    context_budget_from_config,
    run_agent_turn,
    SessionStartRequest,
    start_agent_application_session,
)
from mycode.presentation.cli.confirmer import TerminalConfirmer
from mycode.presentation.cli.presenter import CliDisplayMode, CliPresenter
from mycode.config import LLMConfig, load_llm_config
from mycode.context.compact import ConversationCompactor
from mycode.context.budget import (
    ContextBudgetExceededError,
    format_model_context_stats,
)
from mycode.error_handling import error_summary, format_model_error
from mycode.conversation import Conversation
from mycode.mcp import (
    MCPConfig,
    MCPConfigError,
    load_mcp_config_layers,
    resolve_project_mcp_trust,
)
from mycode.observability import ObservationSink
from mycode.llm import OpenAICompatibleLLMClient
from mycode.project import ProjectIdentity
from mycode.agent.runner import AgentRunner
from mycode.agent.outcome import AgentRunOutcome
from mycode.session import ChatSession
from mycode.persistence.session_store import (
    SessionInUseError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)
from mycode.presentation.cli.subagent_observer import CliSubAgentObserver
from mycode.presentation.cli.session_menu import select_session_request
from mycode.presentation.cli.mcp_trust import TerminalMCPTrustConfirmer
from mycode.presentation.tui.app import run_tui
from mycode.tools import (
    Workspace,
)


EXIT_COMMANDS = {"/exit", "/quit"}


def build_chat_session(llm_config: LLMConfig | None = None) -> ChatSession:
    config = load_llm_config() if llm_config is None else llm_config
    session_id = uuid4().hex
    client = OpenAICompatibleLLMClient(config=config, session_id=session_id)
    summary_client = OpenAICompatibleLLMClient(
        config=config,
        model=config.compact_model,
        thinking_enabled=False,
        session_id=session_id,
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
    *,
    turn_func: Callable[..., AgentRunOutcome] | None = None,
) -> AgentRunOutcome | None:
    if output_chunk_func is None:
        output_chunk_func = _print_chunk
    if turn_func is None:
        def effective_turn_func(
            content: str,
            *,
            event_handler: Callable[[AgentEvent], None],
        ) -> AgentRunOutcome:
            return run_agent_turn(
                runner,
                content,
                event_handler=event_handler,
            )
    else:
        effective_turn_func = turn_func

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
            turn_func=effective_turn_func,
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
    store = SessionStore() if session_store is None else session_store
    effective_request = session_request
    try:
        if effective_request is None:
            effective_request = select_session_request(
                store,
                project,
                input_func=input_func,
                output_func=output_func,
            )
    except (SessionNotFoundError, SessionInUseError) as error:
        output_func(f"session> 错误：{error}")
        return
    except SessionStoreError as error:
        output_func(f"session> 严重错误：{error}")
        return
    if effective_request is None:
        return

    try:
        if mcp_config is None:
            loaded_mcp_config = load_mcp_config_layers(
                workspace_root=workspace.root
            )
            trust_confirmer = TerminalMCPTrustConfirmer(
                input_func=input_func,
                output_func=output_func,
            )
            trust_resolution = resolve_project_mcp_trust(
                loaded_mcp_config,
                project,
                confirmer=trust_confirmer,
            )
            effective_mcp_config = trust_resolution.config
        else:
            effective_mcp_config = mcp_config
    except MCPConfigError as error:
        output_func("MCP servers:")
        output_func(f"✗ config      {error}")
        effective_mcp_config = MCPConfig()

    confirmer = TerminalConfirmer(
        input_func=input_func,
        output_func=output_func,
    )
    cli_observer = CliSubAgentObserver(output=output_func, mode=display_mode)
    interactive_session_selection = session_request is None
    while True:
        try:
            application_session = start_agent_application_session(
                store,
                project,
                request=effective_request,
                mcp_config=effective_mcp_config,
                confirmer=confirmer,
                external_observer=cli_observer,
                llm_config=config,
                observability_sink=observability_sink,
            )
        except SessionInUseError as error:
            if not interactive_session_selection:
                output_func(f"session> 错误：{error}")
                return
            output_func(f"session> 当前不可用：{error}")
            try:
                effective_request = select_session_request(
                    store,
                    project,
                    input_func=input_func,
                    output_func=output_func,
                )
            except (SessionNotFoundError, SessionInUseError) as menu_error:
                output_func(f"session> 错误：{menu_error}")
                return
            except SessionStoreError as menu_error:
                output_func(f"session> 严重错误：{menu_error}")
                return
            if effective_request is None:
                return
            continue
        except SessionNotFoundError as error:
            output_func(f"session> 错误：{error}")
            return
        except SessionStoreError as error:
            output_func(f"session> 严重错误：{error}")
            return
        except Exception as error:
            output_func("")
            output_func(
                "错误> " + format_model_error(error, operation="Agent 运行失败")
            )
            return
        break

    _output_session_started(application_session, output_func)
    if application_session.compact_state_recovered:
        output_func(
            "session> 警告：无效的 Compact 状态已重置；"
            "已恢复完整历史并进入 Compact 冷却期"
        )
    _output_mcp_statuses(application_session, output_func)
    try:
        run_agent_loop(
            runner=application_session.runner,
            input_func=input_func,
            output_func=output_func,
            output_chunk_func=output_chunk_func,
            display_mode=display_mode,
            turn_func=lambda content, *, event_handler: _run_cli_application_turn(
                application_session,
                content,
                event_handler=event_handler,
            ),
        )
    except KeyboardInterrupt:
        try:
            application_session.interrupt()
        except SessionStoreError as lifecycle_error:
            output_func(f"session> 警告：{lifecycle_error}")
        output_func("")
        output_func("提示> Agent 已中断，当前进度已保存。")
    except Exception as error:
        try:
            application_session.interrupt()
        except SessionStoreError as lifecycle_error:
            output_func(f"session> 警告：{lifecycle_error}")
        output_func("")
        output_func(
            "错误> " + format_model_error(error, operation="Agent 运行失败")
        )
    else:
        application_session.close()
    finally:
        try:
            application_session.interrupt()
        except SessionStoreError as lifecycle_error:
            output_func(f"session> 警告：{lifecycle_error}")


def _output_session_started(
    application_session: AgentApplicationSession,
    output_func: Callable[[str], None],
) -> None:
    active_session = application_session.active_project_session
    if active_session.created:
        output_func(f"session> 已创建新会话 {active_session.record.id}")
    else:
        output_func(
            f"session> 已恢复 {active_session.record.id}："
            f"{active_session.record.title}"
        )


def _output_mcp_statuses(
    application_session: AgentApplicationSession,
    output_func: Callable[[str], None],
) -> None:
    if not application_session.mcp_statuses:
        return
    output_func("MCP servers:")
    for status in application_session.mcp_statuses:
        if status.status == "connected":
            output_func(f"✓ {status.alias:<12} {status.tool_count} tools")
        else:
            output_func(
                f"✗ {status.alias:<12} "
                f"{status.error_summary or status.error_type or 'unavailable'}"
            )


def _run_agent_turn(
    *,
    runner: AgentRunner,
    content: str,
    output_func: Callable[[str], None],
    output_chunk_func: Callable[[str], None],
    presenter: CliPresenter,
    turn_func: Callable[..., AgentRunOutcome],
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

    outcome = turn_func(content, event_handler=show_event)

    if assistant_started:
        output_func("")
    presenter.flush()
    return outcome


def _run_cli_application_turn(
    application_session: AgentApplicationSession,
    content: str,
    *,
    event_handler: Callable[[AgentEvent], None],
) -> AgentRunOutcome:
    def handle_runtime_event(runtime_event: RuntimeEvent) -> None:
        if runtime_event.type == "agent" and runtime_event.agent_event is not None:
            event_handler(runtime_event.agent_event)

    return application_session.run_turn(
        content,
        event_handler=handle_runtime_event,
    )


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

    if options.command == "runtime":
        runtime_request: SessionStartRequest | None = None
        if options.new:
            runtime_request = SessionStartRequest(mode="new")
        elif options.continue_session:
            runtime_request = SessionStartRequest(mode="continue")
        elif options.resume is not None:
            runtime_request = SessionStartRequest(
                mode="resume",
                session_id=options.resume,
            )
        exit_code = run_jsonl_runtime(session_request=runtime_request)
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    if options.command == "tui":
        try:
            run_tui()
        except KeyboardInterrupt:
            print("")
            print("提示> TUI 已中断。")
        except Exception as error:
            print(f"错误> TUI 启动失败：{error_summary(error)}")
        return

    session_request: SessionStartRequest | None = None
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
            "  mycode tui\n"
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

    tui_parser = subparsers.add_parser(
        "tui",
        add_help=False,
        help="启动 Textual 终端用户界面",
        description=(
            "启动 MyCode Textual 终端用户界面。\n"
            "14.6.2 提供 Welcome、Session 启动和主界面展示；不执行 Agent Turn。"
        ),
    )
    _add_help_argument(tui_parser)

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

    runtime_parser = subparsers.add_parser(
        "runtime",
        add_help=False,
        help="启动机器可消费的 JSONL runtime",
        description="启动 version 1 JSONL runtime；stdout 只输出 JSONL。",
    )
    _add_help_argument(runtime_parser)
    runtime_parser.add_argument(
        "--jsonl",
        action="store_true",
        required=True,
        help="启用 JSONL machine protocol",
    )
    runtime_session_options = runtime_parser.add_argument_group("会话选项")
    runtime_session_group = runtime_session_options.add_mutually_exclusive_group()
    runtime_session_group.add_argument(
        "--new",
        action="store_true",
        help="创建新会话",
    )
    runtime_session_group.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="续接最近会话；没有历史会话时新建",
    )
    runtime_session_group.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="续接指定的未删除会话",
    )
    return parser


def _add_help_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="显示此帮助并退出",
    )
