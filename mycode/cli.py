# Imported first on purpose: it starts the startup-profiling clock before the
# heavier CLI dependencies are imported.
from mycode import startup_profile

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from mycode.agent.events import AgentEvent
from mycode.adapters.jsonl import run_jsonl_runtime
from mycode.application.events import RuntimeEvent
from mycode.application import (
    AgentApplicationSession,
    ApplicationStartupWarning,
    CompactResult,
    ContextStatus,
    create_application_environment,
    list_project_sessions,
    prepare_application_session_factory,
    run_agent_turn,
    SessionStartRequest,
)
from mycode.presentation.cli.confirmer import TerminalConfirmer
from mycode.presentation.cli.presenter import CliDisplayMode, CliPresenter
from mycode.config import LLMConfig
from mycode.error_handling import error_summary, format_model_error
from mycode.mcp.config import MCPConfig
from mycode.observability import ObservationSink
from mycode.project import ProjectIdentity
from mycode.agent.runner import AgentRunner
from mycode.agent.outcome import AgentRunOutcome
from mycode.persistence.session_store import (
    SessionInUseError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)
from mycode.presentation.cli.subagent_observer import CliSubAgentObserver
from mycode.presentation.cli.session_menu import select_session_request
from mycode.presentation.cli.mcp_trust import TerminalMCPTrustConfirmer
from mycode.presentation.commands import (
    CommandParseError,
    ParsedCommand,
    parse_slash_command,
)
from mycode.presentation.command_format import (
    format_command_help,
    format_compact_result,
    format_context_status,
    format_session_list,
)
from mycode.presentation.tui.app import run_tui


def run_agent_loop(
    runner: AgentRunner,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    output_chunk_func: Callable[[str], None] | None = None,
    display_mode: CliDisplayMode = "normal",
    *,
    turn_func: Callable[..., AgentRunOutcome] | None = None,
    command_handler: Callable[[ParsedCommand], bool] | None = None,
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

        if content == "":
            continue

        try:
            command = parse_slash_command(content)
        except CommandParseError as error:
            output_func(f"command> {error}")
            continue
        if command is not None:
            if command.name == "exit" and command_handler is None:
                break
            if command_handler is None:
                output_func(f"command> /{command.name} is not available yet.")
                continue
            if command_handler(command):
                break
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
    environment = create_application_environment(
        workspace_path,
        session_store=session_store,
    )
    project = environment.project
    store = environment.session_store
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

    confirmer = TerminalConfirmer(
        input_func=input_func,
        output_func=output_func,
    )
    cli_observer = CliSubAgentObserver(output=output_func, mode=display_mode)
    trust_confirmer = TerminalMCPTrustConfirmer(
        input_func=input_func,
        output_func=output_func,
    )

    def handle_startup_warning(warning: ApplicationStartupWarning) -> None:
        if warning.code == "mcp_config_error":
            output_func("MCP servers:")
            output_func(f"✗ config      {warning.message}")

    try:
        factory = prepare_application_session_factory(
            environment,
            llm_config=llm_config,
            mcp_config=mcp_config,
            mcp_trust_confirmer=trust_confirmer,
            confirmer=confirmer,
            external_observer=cli_observer,
            observability_sink=observability_sink,
            warning_handler=handle_startup_warning,
        )
    except Exception as error:
        output_func("")
        output_func(
            "错误> " + format_model_error(error, operation="Agent 启动失败")
        )
        return

    interactive_session_selection = session_request is None
    while True:
        try:
            application_session = factory.open_session(effective_request)
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

    startup_profile.total("runtime.ready")
    _output_session_started(application_session, output_func)
    if application_session.compact_state_recovered:
        output_func(
            "session> 警告：无效的 Compact 状态已重置；"
            "已恢复完整历史并进入 Compact 冷却期"
        )
    _output_mcp_statuses(application_session, output_func)

    def handle_command(command: ParsedCommand) -> bool:
        nonlocal application_session

        if command.name == "help":
            _output_command_help(output_func)
            return False
        if command.name == "sessions":
            _output_sessions(
                store,
                project,
                current_session_id=application_session.active_project_session.record.id,
                output_func=output_func,
            )
            return False
        if command.name == "context":
            try:
                status = application_session.get_context_status()
            except Exception as error:  # noqa: BLE001 - command boundary
                output_func(
                    "context> failed: "
                    + format_model_error(error, operation="Context inspection failed")
                )
            else:
                _output_context_status(status, output_func)
            return False
        if command.name == "compact":
            _output_compact_result(application_session.compact_context(), output_func)
            return False
        if command.name == "exit":
            return True
        if command.name not in {"new", "resume"}:
            output_func(f"command> /{command.name} is not available yet.")
            return False

        if command.name == "resume":
            target_session_id = command.args[0]
            current_session_id = application_session.active_project_session.record.id
            if target_session_id == current_session_id:
                output_func("session> already using current session")
                return False
            request = SessionStartRequest(
                mode="resume",
                session_id=target_session_id,
            )
        else:
            request = SessionStartRequest(mode="new")

        try:
            replacement = factory.open_session(request)
        except SessionInUseError as error:
            output_func(f"session> 当前不可用：{error}")
            return False
        except SessionNotFoundError as error:
            output_func(f"session> 错误：{error}")
            return False
        except SessionStoreError as error:
            output_func(f"session> 严重错误：{error}")
            return False
        except Exception as error:
            output_func(
                "错误> " + format_model_error(error, operation="Session 切换失败")
            )
            return False

        previous = application_session
        application_session = replacement
        try:
            previous.close()
        except Exception as error:
            output_func(
                "session> 警告："
                + format_model_error(error, operation="旧 Session 清理失败")
            )

        _output_session_started(application_session, output_func)
        _output_mcp_statuses(application_session, output_func)
        return False

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
            command_handler=handle_command,
        )
    except KeyboardInterrupt:
        _try_cleanup_application_session(
            application_session.interrupt,
            output_func=output_func,
        )
        output_func("")
        output_func("提示> Agent 已中断，当前进度已保存。")
    except Exception as error:
        _try_cleanup_application_session(
            application_session.interrupt,
            output_func=output_func,
        )
        output_func("")
        output_func(
            "错误> " + format_model_error(error, operation="Agent 运行失败")
        )
    else:
        _try_cleanup_application_session(
            application_session.close,
            output_func=output_func,
        )
    finally:
        _try_cleanup_application_session(
            application_session.interrupt,
            output_func=output_func,
        )


def _try_cleanup_application_session(
    cleanup: Callable[[], None],
    *,
    output_func: Callable[[str], None],
) -> None:
    """Report a lifecycle cleanup failure without replacing the primary result."""
    try:
        cleanup()
    except Exception as lifecycle_error:  # noqa: BLE001 - CLI cleanup boundary
        output_func(
            "session> 警告："
            + format_model_error(lifecycle_error, operation="Session 清理失败")
        )


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


def _output_command_help(output_func: Callable[[str], None]) -> None:
    for line in format_command_help():
        output_func(f"command> {line}")


def _output_context_status(
    status: ContextStatus,
    output_func: Callable[[str], None],
) -> None:
    for line in format_context_status(status):
        output_func(f"command> {line}")


def _output_compact_result(
    result: CompactResult,
    output_func: Callable[[str], None],
) -> None:
    output_func(f"context> {format_compact_result(result)}")


def _output_sessions(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    current_session_id: str,
    output_func: Callable[[str], None],
) -> None:
    try:
        sessions = list_project_sessions(store, project, limit=10)
    except SessionStoreError as error:
        output_func(f"session> 严重错误：{error}")
        return

    for line in format_session_list(
        sessions,
        current_session_id=current_session_id,
    ):
        output_func(f"command> {line}")


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
    startup_profile.total("cli.import")
    args = sys.argv[1:] if argv is None else argv

    if args == []:
        print("MyCode 已就绪。")
        return

    parser = _build_cli_parser()
    options = parser.parse_args(args)

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
        description="可扩展终端 coding agent，以启动命令时的当前目录作为工作区。",
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

    tui_parser = subparsers.add_parser(
        "tui",
        add_help=False,
        help="启动 Textual 终端用户界面",
        description="启动 MyCode Textual TUI，支持 Session、Agent Turn、Permission 和交互式命令。",
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
