from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from my_agent.plan.types import PlanState
from my_agent.text_safety import sanitize_json_value


class PlanStore(Protocol):
    def save(self, plan: PlanState) -> None:
        ...

    def get(self, plan_id: str) -> PlanState | None:
        ...


class InMemoryPlanStore:
    def __init__(self) -> None:
        self._plans: dict[str, PlanState] = {}

    def save(self, plan: PlanState) -> None:
        self._plans[plan.id] = PlanState.from_dict(plan.to_dict())

    def get(self, plan_id: str) -> PlanState | None:
        plan = self._plans.get(plan_id)
        return PlanState.from_dict(plan.to_dict()) if plan is not None else None


class JsonPlanStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, plan: PlanState) -> None:
        path = self._path_for(plan.id)
        path.write_text(json.dumps(sanitize_json_value(plan.to_dict()), ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, plan_id: str) -> PlanState | None:
        path = self._path_for(plan_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return PlanState.from_dict(payload)

    def _path_for(self, plan_id: str) -> Path:
        safe_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in plan_id)
        return self.directory / f"{safe_id}.json"

