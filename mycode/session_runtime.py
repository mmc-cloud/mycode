from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import math
from pathlib import Path
from threading import Event, Thread
from typing import Literal
from uuid import uuid4

from mycode.artifacts import (
    ArtifactCleanupError,
    artifact_directory_for_session,
    validate_session_artifact_cleanup_target,
)
from mycode.context_compact import CompactState
from mycode.conversation import Conversation
from mycode.messages import Message
from mycode.project import ProjectIdentity
from mycode.session_deletion import SessionDeletionManager
from mycode.session_lock import SessionLockError
from mycode.session_store import (
    DEFAULT_SESSION_TITLE,
    DEFAULT_SESSION_LEASE_SECONDS,
    SessionInUseError,
    SessionDatabaseCorruptionError,
    SessionDeletingError,
    SessionLeaseLostError,
    SessionNotFoundError,
    SessionRecord,
    SessionStore,
    SessionStoreError,
)


SessionStartMode = Literal["select", "new", "continue", "resume"]
DEFAULT_SESSION_LIST_LIMIT = 10
AUTO_SESSION_TITLE_CHARS = 80
SESSION_LEASE_HEARTBEAT_SECONDS = DEFAULT_SESSION_LEASE_SECONDS / 3


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
    record: SessionRecord
    lease_owner_id: str
    lease_duration_seconds: float = DEFAULT_SESSION_LEASE_SECONDS
    heartbeat_interval_seconds: float = SESSION_LEASE_HEARTBEAT_SECONDS
    _heartbeat_stop: Event = field(default_factory=Event, init=False, repr=False)
    _heartbeat_thread: Thread | None = field(default=None, init=False, repr=False)
    _lease_error: SessionLeaseLostError | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _compact_state_recovered: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.lease_duration_seconds)
            or self.lease_duration_seconds <= 0
        ):
            raise ValueError("lease_duration_seconds must be above 0.")
        if (
            not math.isfinite(self.heartbeat_interval_seconds)
            or self.heartbeat_interval_seconds <= 0
            or self.heartbeat_interval_seconds >= self.lease_duration_seconds
        ):
            raise ValueError(
                "heartbeat_interval_seconds must be above 0 and below the lease duration."
            )

    def load_history(self) -> Conversation:
        return self.store.load_conversation(self.project, self.record.id)

    def load_compact_state(self) -> CompactState:
        loaded = self.store.load_or_reset_compact_state(
            self.project,
            self.record.id,
            lease_owner_id=self.lease_owner_id,
        )
        self._compact_state_recovered = loaded.recovered_invalid_state
        return loaded.state

    @property
    def compact_state_recovered(self) -> bool:
        return self._compact_state_recovered

    @property
    def artifact_directory(self) -> Path:
        return artifact_directory_for_session(
            self.store.database_path.parent,
            project_key=self.project.key,
            session_id=self.record.id,
        )

    def artifact_write_guard(self) -> AbstractContextManager[None]:
        return self.store.artifact_write_guard(
            self.project,
            self.record.id,
            self.lease_owner_id,
        )

    def persist_message(self, message: Message) -> None:
        self._raise_if_lease_lost()
        self.store.append_message(
            self.project,
            self.record.id,
            message,
            lease_owner_id=self.lease_owner_id,
        )
        if message.role == "user" and self.record.title == DEFAULT_SESSION_TITLE:
            title = _session_title_from_message(message.content)
            if title is not None:
                self.record = self.store.rename_session(
                    self.project,
                    self.record.id,
                    title,
                    lease_owner_id=self.lease_owner_id,
                )

    def persist_compact_state(self, state: CompactState) -> None:
        self._raise_if_lease_lost()
        self.store.save_compact_state(
            self.project,
            self.record.id,
            state,
            lease_owner_id=self.lease_owner_id,
        )

    def start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return
        thread = Thread(
            target=self._heartbeat_loop,
            name=f"mycode-session-{self.record.id[:8]}",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            self._heartbeat_thread = None
            raise
        self._heartbeat_thread = thread

    def close(self) -> None:
        self._stop_heartbeat()
        self._raise_if_lease_lost()
        self.record = self.store.release_session_lease(
            self.project,
            self.record.id,
            self.lease_owner_id,
            "closed",
        )

    def interrupt(self) -> None:
        self._stop_heartbeat()
        self._raise_if_lease_lost()
        self.record = self.store.release_session_lease(
            self.project,
            self.record.id,
            self.lease_owner_id,
            "interrupted",
        )

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
            try:
                self.store.renew_session_lease(
                    self.project,
                    self.record.id,
                    self.lease_owner_id,
                    lease_duration_seconds=self.lease_duration_seconds,
                )
            except (
                SessionDeletingError,
                SessionLeaseLostError,
                SessionNotFoundError,
            ) as error:
                self._lease_error = SessionLeaseLostError(str(error))
                return
            except SessionStoreError:
                continue

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None:
            thread.join(timeout=self.store.busy_timeout_seconds + 1)

    def _raise_if_lease_lost(self) -> None:
        if self._lease_error is not None:
            raise self._lease_error


def start_project_session(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    request: SessionStartRequest | None = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> ActiveProjectSession | None:
    effective_request = SessionStartRequest() if request is None else request
    _retry_pending_session_deletions(store, output_func)

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
    store.expire_session_leases(project)
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
        validate_session_artifact_cleanup_target(
            store.database_path.parent,
            project_key=project.key,
            session_id=target.id,
        )
    except ArtifactCleanupError as error:
        output_func(f"session> 删除被阻止：{error}")
        return

    try:
        result = SessionDeletionManager(store).request_and_process(
            project,
            target.id,
        )
    except SessionInUseError as error:
        output_func(f"session> 当前无法删除：{error}")
        return
    except SessionLockError as error:
        output_func(f"session> 删除清理待处理：{error}")
        return
    except SessionDatabaseCorruptionError:
        raise
    except SessionStoreError as error:
        output_func(f"session> 删除失败：{error}")
        return

    if not result.completed:
        output_func(
            "session> 已记录删除请求，物理清理仍待完成："
            f"stage={result.pending_stage}, reason={result.error_code}, "
            f"retries={result.retry_count}"
        )
        return

    if result.already_absent:
        output_func(f"session> 会话此前已经删除：{target.id}")
    else:
        output_func(f"session> 已永久删除 {target.id}：{target.title}")
    if result.artifact_removed:
        output_func("session> artifact 清理完成")
    else:
        output_func("session> 没有需要清理的 artifact 文件")


def _retry_pending_session_deletions(
    store: SessionStore,
    output_func: Callable[[str], None],
) -> None:
    try:
        results = SessionDeletionManager(store).retry_all_pending()
    except SessionDatabaseCorruptionError:
        raise
    except (SessionLockError, SessionStoreError) as error:
        output_func(f"session> 警告：无法重试待处理的删除任务：{error}")
        return

    for result in results:
        if result.maintenance_only:
            if not result.completed:
                output_func(
                    "session> 警告：匿名数据库清理仍待完成："
                    f"stage={result.pending_stage}, "
                    f"reason={result.error_code}, retries={result.retry_count}"
                )
            continue
        if result.completed:
            output_func(
                "session> 已完成待处理的删除任务："
                f"project={_short_project_key(result.project_key)}, "
                f"session={result.session_id}"
            )
            continue
        output_func(
            "session> 警告：删除清理仍待完成："
            f"project={_short_project_key(result.project_key)}, "
            f"session={result.session_id}, stage={result.pending_stage}, "
            f"reason={result.error_code}, retries={result.retry_count}"
        )


def _short_project_key(project_key: str | None) -> str:
    if project_key is None:
        return "未知"
    return project_key[:8]


def _create_session(
    store: SessionStore,
    project: ProjectIdentity,
    output_func: Callable[[str], None],
) -> ActiveProjectSession:
    owner_id = str(uuid4())
    record = store.create_session(
        project,
        lease_owner_id=owner_id,
        lease_duration_seconds=DEFAULT_SESSION_LEASE_SECONDS,
    )
    output_func(f"session> 已创建新会话 {record.id}")
    return ActiveProjectSession(
        store=store,
        project=project,
        record=record,
        lease_owner_id=owner_id,
    )


def _resume_session(
    store: SessionStore,
    project: ProjectIdentity,
    record: SessionRecord,
    output_func: Callable[[str], None],
) -> ActiveProjectSession:
    owner_id = str(uuid4())
    active_record = store.acquire_session_lease(
        project,
        record.id,
        owner_id,
        lease_duration_seconds=DEFAULT_SESSION_LEASE_SECONDS,
    )
    output_func(f"session> 已恢复 {active_record.id}：{active_record.title}")
    return ActiveProjectSession(
        store=store,
        project=project,
        record=active_record,
        lease_owner_id=owner_id,
    )


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
