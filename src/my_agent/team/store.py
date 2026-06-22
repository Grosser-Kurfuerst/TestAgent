from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from my_agent.team.types import TeamState
from my_agent.text_safety import sanitize_json_value


class TeamStore(Protocol):
    def save(self, team: TeamState) -> None:
        ...

    def get(self, team_id: str) -> TeamState | None:
        ...


class InMemoryTeamStore:
    def __init__(self) -> None:
        self._teams: dict[str, TeamState] = {}

    def save(self, team: TeamState) -> None:
        self._teams[team.id] = TeamState.from_dict(team.to_dict())

    def get(self, team_id: str) -> TeamState | None:
        team = self._teams.get(team_id)
        return TeamState.from_dict(team.to_dict()) if team is not None else None


class JsonTeamStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, team: TeamState) -> None:
        path = self._path_for(team.id)
        path.write_text(json.dumps(sanitize_json_value(team.to_dict()), ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, team_id: str) -> TeamState | None:
        path = self._path_for(team_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return TeamState.from_dict(payload)

    def _path_for(self, team_id: str) -> Path:
        safe_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in team_id)
        return self.directory / f"{safe_id}.json"

