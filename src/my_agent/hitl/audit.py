from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    timestamp: str
    request_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments_summary: dict[str, Any]
    risk_level: str
    policy_decision: str
    approval_decision: str
    approver: str
    outcome: str
    reason: str
    elapsed_ms: int
    server_name: str = ""


@dataclass(frozen=True)
class AuditRecordResult:
    ok: bool
    path: Path | None = None
    error: str = ""


class AuditLog:
    def __init__(self, audit_dir: str | Path | None = None) -> None:
        self.audit_dir = Path(audit_dir).expanduser() if audit_dir is not None else Path("~/.agentcli/audit").expanduser()
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Any | None) -> "AuditLog":
        return cls(getattr(config, "hitl_audit_dir", None))

    def record(self, entry: AuditEntry) -> AuditRecordResult:
        path = self.audit_dir / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
        except OSError as exc:
            return AuditRecordResult(ok=False, path=path, error=f"{type(exc).__name__}: {exc}")
        return AuditRecordResult(ok=True, path=path)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
