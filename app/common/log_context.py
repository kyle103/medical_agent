from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass
from typing import Optional


_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


@dataclass(frozen=True)
class RequestContext:
    trace_id: str
    request_id: str


def new_trace_id() -> str:
    return uuid.uuid4().hex


def set_request_context(*, trace_id: Optional[str] = None, request_id: Optional[str] = None) -> RequestContext:
    t_id = trace_id or new_trace_id()
    r_id = request_id or uuid.uuid4().hex
    _trace_id_var.set(t_id)
    _request_id_var.set(r_id)
    return RequestContext(trace_id=t_id, request_id=r_id)


def get_trace_id() -> str:
    return _trace_id_var.get()


def get_request_id() -> str:
    return _request_id_var.get()


def clear_request_context() -> None:
    _trace_id_var.set("-")
    _request_id_var.set("-")
