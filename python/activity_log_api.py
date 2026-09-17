"""Register activity-log routes on the existing Arduino WebUI instance."""

from __future__ import annotations

from typing import Any

from activity_log import ActivityLog

activity_log = ActivityLog()


def _safe(callable_, *args: Any) -> dict[str, Any]:
    try:
        return callable_(*args)
    except ValueError as exc:
        # Malformed/oversized input: retrying the same batch can never succeed.
        return {"ok": False, "error": str(exc), "retryable": False}
    except Exception as exc:
        # Locked/full disk etc.: the browser keeps the batch and retries later.
        return {"ok": False, "error": str(exc), "retryable": True}


def receive_activity(payload: str = "") -> dict[str, Any]:
    return _safe(activity_log.ingest_json, payload)


def activity_status() -> dict[str, Any]:
    return _safe(activity_log.status)


def register_activity_log_routes(web_ui, routes_include_api_prefix: bool = True) -> ActivityLog:
    """Attach the browser-ingest route and a diagnostic status route.

    Leave routes_include_api_prefix=True when existing routes are registered as
    '/api/data'. Set it to False when WebUI(api_path_prefix='/api') is used and
    existing routes are registered as '/data'.
    """
    prefix = "/api" if routes_include_api_prefix else ""
    web_ui.expose_api("GET", f"{prefix}/activity-log", receive_activity)
    web_ui.expose_api("GET", f"{prefix}/activity-status", activity_status)
    return activity_log
