import os
import signal
import subprocess


class ProcessTreeCleanup:
    def __init__(
        self,
        *,
        method: str,
        job_handle: int | None = None,
        process_group_id: int | None = None,
        error: str | None = None,
    ) -> None:
        self.method = method
        self.error = error
        self._job_handle = job_handle
        self._process_group_id = process_group_id
        self._closed = False
        self._success = error is None

    def close(self) -> bool:
        if self._closed:
            return self._success

        self._closed = True
        if self._job_handle is not None:
            self._success = _close_windows_job_handle(self._job_handle)
            if not self._success:
                self.error = self.error or "CloseHandle failed for process job."
            return self._success

        if self._process_group_id is not None:
            self._success = _kill_posix_process_group(self._process_group_id)
            if not self._success:
                self.error = self.error or "Failed to terminate process group."
            return self._success

        return self._success


def process_tree_popen_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {}

    return {"start_new_session": True}


def create_process_tree_cleanup(
    process: subprocess.Popen[str],
) -> ProcessTreeCleanup:
    if os.name == "nt":
        return _create_windows_job_cleanup(process)

    return ProcessTreeCleanup(
        method="posix_process_group",
        process_group_id=process.pid,
    )


def _create_windows_job_cleanup(
    process: subprocess.Popen[str],
) -> ProcessTreeCleanup:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _JobObjectBasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _JobObjectExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _JobObjectBasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

        job_handle = kernel32.CreateJobObjectW(None, None)
        if not job_handle:
            raise ctypes.WinError(ctypes.get_last_error())

        limit_info = _JobObjectExtendedLimitInformation()
        limit_info.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            job_handle,
            9,
            ctypes.byref(limit_info),
            ctypes.sizeof(limit_info),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            _close_windows_job_handle(int(job_handle))
            raise error

        process_handle = getattr(process, "_handle")
        if not kernel32.AssignProcessToJobObject(job_handle, process_handle):
            error = ctypes.WinError(ctypes.get_last_error())
            _close_windows_job_handle(int(job_handle))
            raise error

        return ProcessTreeCleanup(method="windows_job", job_handle=int(job_handle))
    except Exception as error:
        return ProcessTreeCleanup(
            method="windows_job",
            error=f"{type(error).__name__}: {error}",
        )


def _close_windows_job_handle(job_handle: int) -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        return bool(kernel32.CloseHandle(wintypes.HANDLE(job_handle)))
    except Exception:
        return False


def _kill_posix_process_group(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False

    return True
