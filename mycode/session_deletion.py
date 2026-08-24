from dataclasses import dataclass

from mycode.artifacts import (
    ArtifactCleanupError,
    delete_quarantined_artifacts,
    quarantine_session_artifacts,
    validate_session_artifact_cleanup_target,
)
from mycode.project import ProjectIdentity
from mycode.session_lock import SessionLockError, SessionLockTimeoutError
from mycode.session_store import (
    DatabaseMaintenanceState,
    SessionDatabaseCorruptionError,
    SessionDeletionRecord,
    SessionMaintenanceError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)


POST_DELETE_SCRUB_STAGE = "post_delete_scrub"


@dataclass(frozen=True)
class SessionDeletionResult:
    session_id: str
    deletion_id: str | None
    completed: bool
    project_key: str | None = None
    already_absent: bool = False
    artifact_removed: bool = False
    pending_stage: str | None = None
    error_code: str | None = None
    retry_count: int = 0
    maintenance_only: bool = False


@dataclass
class SessionDeletionManager:
    store: SessionStore

    def request_and_process(
        self,
        project: ProjectIdentity,
        session_id: str,
    ) -> SessionDeletionResult:
        validate_session_artifact_cleanup_target(
            self.store.database_path.parent,
            project_key=project.key,
            session_id=session_id,
        )
        task = self.store.request_session_deletion(project, session_id)
        if task is None:
            maintenance = self._retry_post_delete_scrub()
            if maintenance is not None and not maintenance.completed:
                return SessionDeletionResult(
                    session_id=session_id,
                    deletion_id=None,
                    completed=False,
                    project_key=project.key,
                    already_absent=True,
                    pending_stage=maintenance.pending_stage,
                    error_code=maintenance.error_code,
                    retry_count=maintenance.retry_count,
                )
            return SessionDeletionResult(
                session_id=session_id,
                deletion_id=None,
                completed=True,
                project_key=project.key,
                already_absent=True,
            )
        return self.process_task(task.id)

    def process_task(
        self,
        deletion_id: str,
    ) -> SessionDeletionResult:
        task = self._get_task(deletion_id)
        if task is None:
            maintenance = self._retry_post_delete_scrub()
            if maintenance is not None:
                return maintenance
            return SessionDeletionResult(
                session_id="unknown",
                deletion_id=deletion_id,
                completed=True,
                already_absent=True,
            )

        prepared, failed = self._prepare_task(task)
        if failed is not None:
            return failed
        if prepared is None:
            maintenance = self._retry_post_delete_scrub()
            if maintenance is not None:
                return maintenance
            return SessionDeletionResult(
                session_id=task.session_id,
                deletion_id=task.id,
                completed=True,
                project_key=task.project_key,
                already_absent=True,
                artifact_removed=bool(task.artifact_present),
            )

        with self.store.database_maintenance_lock():
            self.store.retire_session_deletion_for_scrub(prepared.id)
            scrub = self._process_post_delete_scrub_locked()
        return _result_for_scrub(prepared, scrub)

    def retry_all_pending(self) -> list[SessionDeletionResult]:
        results: list[SessionDeletionResult] = []
        prepared_tasks: list[SessionDeletionRecord] = []
        for task in self.store.list_all_session_deletions():
            current = self._get_task(task.id)
            if current is None:
                continue
            prepared, failed = self._prepare_task(current)
            if failed is not None:
                results.append(failed)
            elif prepared is not None:
                prepared_tasks.append(prepared)

        state = self.store.load_database_maintenance_state()
        if not prepared_tasks and not state.post_delete_scrub_required:
            return results

        with self.store.database_maintenance_lock():
            retired: list[SessionDeletionRecord] = []
            for task in prepared_tasks:
                if self.store.retire_session_deletion_for_scrub(task.id):
                    retired.append(task)
            scrub = self._process_post_delete_scrub_locked()

        results.extend(_result_for_scrub(task, scrub) for task in retired)
        if not retired:
            results.append(scrub)
        return results

    def _prepare_task(
        self,
        task: SessionDeletionRecord,
    ) -> tuple[SessionDeletionRecord | None, SessionDeletionResult | None]:
        try:
            with self.store.session_operation_lock_for_key(
                task.project_key,
                task.session_id,
            ):
                current = self._get_task(task.id)
                if current is None:
                    return None, None
                if current.stage == "pending":
                    quarantine = quarantine_session_artifacts(
                        self.store.database_path.parent,
                        project_key=current.project_key,
                        session_id=current.session_id,
                        deletion_id=current.id,
                    )
                    current = self.store.delete_session_records_for_deletion_locked(
                        current.id,
                        artifact_present=quarantine.had_artifacts,
                    )
                    task = current

                # Schema-v5 tasks may be at any later legacy maintenance stage.
                # Their database rows are already gone, so only artifact cleanup
                # remains before replacing the target-bearing task with the
                # anonymous schema-v6 scrub marker.
                delete_quarantined_artifacts(
                    self.store.database_path.parent,
                    deletion_id=current.id,
                )
                return current, None
        except SessionDatabaseCorruptionError:
            raise
        except (
            ArtifactCleanupError,
            OSError,
            SessionLockError,
            SessionMaintenanceError,
            SessionStoreError,
        ) as error:
            error_code = _deletion_error_code(error, stage=task.stage)
            current = self._get_task(task.id)
            if current is None:
                raise
            failed = self.store.record_session_deletion_failure(
                task.id,
                error_code=error_code,
            )
            return None, _pending_task_result(failed)

    def _retry_post_delete_scrub(self) -> SessionDeletionResult | None:
        state = self.store.load_database_maintenance_state()
        if not state.post_delete_scrub_required:
            return None
        with self.store.database_maintenance_lock():
            return self._process_post_delete_scrub_locked()

    def _process_post_delete_scrub_locked(self) -> SessionDeletionResult:
        state = self.store.load_database_maintenance_state()
        if not state.post_delete_scrub_required:
            return _maintenance_result(state, completed=True)

        try:
            if self.store.has_active_session_leases_for_maintenance():
                state = self.store.record_post_delete_scrub_failure(
                    "active_session_leases"
                )
                return _maintenance_result(state, completed=False)

            self.store.checkpoint_wal_truncate()
            self.store.vacuum_database()
            self.store.checkpoint_wal_truncate()
            self.store.mark_post_delete_scrub_complete()
            # Clearing the marker is itself a WAL write. This final checkpoint
            # is part of completion, not best-effort cleanup.
            self.store.checkpoint_wal_truncate()
            state = self.store.load_database_maintenance_state()
            return _maintenance_result(state, completed=True)
        except SessionDatabaseCorruptionError:
            raise
        except (OSError, SessionMaintenanceError, SessionStoreError) as error:
            state = self.store.record_post_delete_scrub_failure(
                _maintenance_error_code(error)
            )
            return _maintenance_result(state, completed=False)

    def _get_task(
        self,
        deletion_id: str,
    ) -> SessionDeletionRecord | None:
        return self.store.get_session_deletion_by_id(deletion_id)


def _result_for_scrub(
    task: SessionDeletionRecord,
    scrub: SessionDeletionResult,
) -> SessionDeletionResult:
    return SessionDeletionResult(
        session_id=task.session_id,
        deletion_id=task.id,
        completed=scrub.completed,
        project_key=task.project_key,
        artifact_removed=bool(task.artifact_present),
        pending_stage=scrub.pending_stage,
        error_code=scrub.error_code,
        retry_count=scrub.retry_count,
    )


def _pending_task_result(task: SessionDeletionRecord) -> SessionDeletionResult:
    return SessionDeletionResult(
        session_id=task.session_id,
        deletion_id=task.id,
        completed=False,
        project_key=task.project_key,
        artifact_removed=bool(task.artifact_present),
        pending_stage=task.stage,
        error_code=task.last_error_code,
        retry_count=task.retry_count,
    )


def _maintenance_result(
    state: DatabaseMaintenanceState,
    *,
    completed: bool,
) -> SessionDeletionResult:
    return SessionDeletionResult(
        session_id="database-maintenance",
        deletion_id=None,
        completed=completed,
        pending_stage=None if completed else POST_DELETE_SCRUB_STAGE,
        error_code=state.last_error_code,
        retry_count=state.retry_count,
        maintenance_only=True,
    )


def _maintenance_error_code(error: Exception) -> str:
    if isinstance(error, SessionMaintenanceError):
        return "sqlite_maintenance_pending"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, OSError):
        return "filesystem_cleanup_failed"
    return "session_store_error"


def _deletion_error_code(error: Exception, *, stage: str) -> str:
    if isinstance(error, SessionLockTimeoutError):
        return "session_lock_timeout"
    if isinstance(error, ArtifactCleanupError):
        if stage == "pending":
            return "artifact_quarantine_failed"
        return "artifact_cleanup_failed"
    if isinstance(error, SessionMaintenanceError):
        return "sqlite_maintenance_pending"
    if isinstance(error, SessionNotFoundError):
        return "deletion_task_missing"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, OSError):
        return "filesystem_cleanup_failed"
    if isinstance(error, SessionStoreError):
        return "session_store_error"
    return "deletion_step_failed"
