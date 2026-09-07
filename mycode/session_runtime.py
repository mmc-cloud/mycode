from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from mycode.context_compact import CompactState
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.session_store import (
    DEFAULT_SESSION_TITLE, SessionInUseError, SessionNotFoundError,
    SessionRecord, SessionStore, SessionStoreError, WritableSession,
)

SessionStartMode = Literal["select", "new", "continue", "resume"]
DEFAULT_SESSION_LIST_LIMIT = 10
AUTO_SESSION_TITLE_CHARS = 80


@dataclass(frozen=True)
class SessionStartRequest:
    mode: SessionStartMode = "select"
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"select", "new", "continue", "resume"}:
            raise ValueError(f"Unsupported session start mode: {self.mode}")
        if self.mode == "resume":
            if self.session_id is None or self.session_id.strip() == "":
                raise ValueError("resume mode requires session_id.")
            return
        if self.session_id is not None:
            raise ValueError(f"{self.mode} mode must not include session_id.")


@dataclass
class ActiveProjectSession:
    store: SessionStore
    project: ProjectIdentity
    writer: WritableSession
    _ownership: AbstractContextManager[WritableSession] = field(repr=False)
    _finished: bool = field(default=False, init=False)
    _compact_state_recovered: bool = field(default=False, init=False)

    @property
    def record(self) -> SessionRecord:
        return replace(
            self.writer.record, status=self.writer.record.status if self._finished else "active",
        )

    def load_history(self) -> Conversation:
        return self.writer.load_history()

    def load_compact_state(self) -> CompactState:
        loaded = self.writer.load_or_reset_compact_state()
        self._compact_state_recovered = loaded.recovered_invalid_state
        return loaded.state

    @property
    def compact_state_recovered(self) -> bool:
        return self._compact_state_recovered

    @property
    def artifact_directory(self) -> Path:
        return self.writer.layout.artifacts_directory

    def persist_message(self, message: Message) -> None:
        self.writer.append_messages([message])
        if message.role == "user" and self.record.title == DEFAULT_SESSION_TITLE:
            title = _session_title_from_message(message.content)
            if title is not None:
                self.writer.rename(title)

    def persist_compact_state(self, state: CompactState) -> None:
        self.writer.save_compact_state(state)

    def close(self) -> None:
        self._finish("closed")

    def interrupt(self) -> None:
        self._finish("interrupted")

    def _finish(self, status: Literal["closed", "interrupted"]) -> None:
        if self._finished:
            return
        try:
            self.writer.finish(status)
        finally:
            self._finished = True
            self._ownership.__exit__(None, None, None)


def start_project_session(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    request: SessionStartRequest | None = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> ActiveProjectSession | None:
    effective_request = SessionStartRequest() if request is None else request

    if effective_request.mode == "new":
        return _create_session(store, project, output_func)

    if effective_request.mode == "continue":
        sessions = store.list_sessions(project, limit=1)
        if not sessions:
            return _create_session(store, project, output_func)
        return _resume_session(store, project, sessions[0], output_func)

    if effective_request.mode == "resume":
        identifier = effective_request.session_id or ""
        session = store.get_session(project, identifier)
        if session is None:
            raise SessionNotFoundError(
                f"Session not found in current project: {identifier}"
            )
        return _resume_session(store, project, session, output_func)

    return _select_session(
        store,
        project,
        input_func=input_func,
        output_func=output_func,
    )


def _select_session(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
) -> ActiveProjectSession | None:
    sessions = store.list_sessions(project, limit=DEFAULT_SESSION_LIST_LIMIT)
    output_func(f"session> 项目 {project.workspace_root}")
    if not sessions:
        output_func("session> 没有历史会话，正在创建新会话")
        return _create_session(store, project, output_func)

    while True:
        if sessions:
            output_func("session> 请选择历史会话，或管理已有会话")
            _output_numbered_sessions(sessions, output_func)
        else:
            output_func("session> 当前没有历史会话")
        output_func("session> [N] 创建新会话")
        if sessions:
            output_func("session> [D] 永久删除会话")
        output_func("session> [Q] 退出")

        try:
            answer = input_func("session> ").strip()
        except EOFError:
            output_func("")
            return None
        normalized = answer.casefold()
        if normalized in {"n", "new"}:
            return _create_session(store, project, output_func)
        if normalized in {"d", "delete"} and sessions:
            _delete_session_interactively(
                store,
                project,
                sessions,
                input_func=input_func,
                output_func=output_func,
            )
            sessions = store.list_sessions(
                project,
                limit=DEFAULT_SESSION_LIST_LIMIT,
            )
            continue
        if normalized in {"q", "quit"}:
            return None
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(sessions):
                try:
                    return _resume_session(
                        store,
                        project,
                        sessions[index - 1],
                        output_func,
                    )
                except SessionInUseError as error:
                    output_func(f"session> 当前不可用：{error}")
                    continue
        output_func("session> 选择无效，请重新输入")


def _output_numbered_sessions(
    sessions: list[SessionRecord],
    output_func: Callable[[str], None],
) -> None:
    for index, session in enumerate(sessions, start=1):
        output_func(
            f"session> [{index}] {session.title} "
            f"({_session_status_label(session.status)}, {session.id[:8]})"
        )


def _delete_session_interactively(
    store: SessionStore,
    project: ProjectIdentity,
    sessions: list[SessionRecord],
    *,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
) -> None:
    output_func("session> 请选择要永久删除的会话")
    _output_numbered_sessions(sessions, output_func)
    output_func("session> [Q] 取消删除")

    target: SessionRecord | None = None
    while target is None:
        try:
            answer = input_func("session delete> ").strip()
        except EOFError:
            output_func("")
            output_func("session> 已取消删除")
            return
        if answer.casefold() in {"q", "quit", "cancel"}:
            output_func("session> 已取消删除")
            return
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(sessions):
                target = sessions[index - 1]
                break
        output_func("session> 删除选项无效，请重新输入")

    confirmation = f"DELETE {target.id}"
    output_func(
        f"session> 将永久删除 {target.title} "
        f"({_session_status_label(target.status)}, {target.id})"
    )
    output_func(
        "session> 这会删除该会话的消息、工具调用、Compact 状态、"
        "SubAgent 记录和 artifact"
    )
    try:
        answer = input_func(
            f"session> 输入 {confirmation} 进行确认："
        ).strip()
    except EOFError:
        output_func("")
        output_func("session> 已取消删除")
        return
    if answer != confirmation:
        output_func("session> 已取消删除：确认文本不匹配")
        return

    try:
        removed = store.delete_session(project, target.id)
    except SessionStoreError as error:
        output_func(f"session> 当前无法删除：{error}")
        return
    if removed:
        output_func(f"session> 已永久删除 {target.id}：{target.title}")
    else:
        output_func(f"session> 会话此前已经删除：{target.id}")


def _open_active(store, project, output_func, *, record=None):
    ownership = store.open_session(
        project, None if record is None else record.id, create=record is None,
    )
    writer = ownership.__enter__()
    active = ActiveProjectSession(store, project, writer, ownership)
    try:
        writer.finish("interrupted")
        if record is None:
            output_func(f"session> 已创建新会话 {active.record.id}")
        else:
            output_func(f"session> 已恢复 {active.record.id}：{active.record.title}")
        return active
    except BaseException:
        active.interrupt()
        raise


def _create_session(store, project, output_func) -> ActiveProjectSession:
    return _open_active(store, project, output_func)


def _resume_session(store, project, record, output_func) -> ActiveProjectSession:
    return _open_active(store, project, output_func, record=record)


def _session_title_from_message(content: str) -> str | None:
    normalized = " ".join(content.split())
    if normalized == "":
        return None
    if len(normalized) <= AUTO_SESSION_TITLE_CHARS:
        return normalized
    return normalized[: AUTO_SESSION_TITLE_CHARS - 3].rstrip() + "..."


def _session_status_label(status: str) -> str:
    return {
        "active": "进行中",
        "closed": "已关闭",
        "interrupted": "已中断",
    }.get(status, status)
