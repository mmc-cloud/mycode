"""Shared busy boundary for model requests and atomic configuration changes."""
from contextlib import contextmanager
from functools import wraps
from threading import RLock


class ModelActivity:
    def __init__(self, validate=None):
        self._lock = RLock()
        self._requests = 0
        self._changing = False
        self._validate = validate

    @contextmanager
    def operation(self, role=None):
        with self._lock:
            if self._changing:
                raise RuntimeError("Model configuration is changing; wait for completion.")
            if self._validate is not None and role is not None:
                self._validate(role)
            self._requests += 1
        try:
            yield
        finally:
            with self._lock:
                self._requests -= 1

    @contextmanager
    def change(self):
        with self._lock:
            if self._requests or self._changing:
                raise RuntimeError("Model requests are active; wait before changing configuration.")
            self._changing = True
        try:
            yield
        finally:
            with self._lock:
                self._changing = False


def model_operation(role):
    def decorate(method):
        @wraps(method)
        def guarded(self, *args, **kwargs):
            activity = getattr(self, "model_activity", None)
            if activity is None:
                return method(self, *args, **kwargs)
            with activity.operation(role):
                return method(self, *args, **kwargs)
        return guarded
    return decorate
