from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_agent.tools.spec import ToolContext, ToolRegistration, ToolSource


@dataclass(frozen=True)
class PluginToolSource(ToolSource):
    repo_root: Path
    config: Any | None = None
    name: str = "plugin"

    def load(self, context: ToolContext) -> list[ToolRegistration]:
        registrations: list[ToolRegistration] = []
        registrations.extend(self._load_entry_points(context))
        if bool(getattr(self.config, "enable_project_plugins", False)):
            registrations.extend(self._load_project_plugins(context))
        return registrations

    def _load_entry_points(self, context: ToolContext) -> list[ToolRegistration]:
        registrations: list[ToolRegistration] = []
        entry_points = importlib.metadata.entry_points()
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group="my_agent.tools")
        else:
            selected = entry_points.get("my_agent.tools", [])  # type: ignore[assignment]
        for entry_point in selected:
            factory = entry_point.load()
            registrations.extend(_coerce_registrations(factory(context), f"entry point {entry_point.name}"))
        return registrations

    def _load_project_plugins(self, context: ToolContext) -> list[ToolRegistration]:
        plugin_dir = self.repo_root / ".agentcli" / "plugins"
        if not plugin_dir.exists():
            return []
        registrations: list[ToolRegistration] = []
        for manifest in sorted(plugin_dir.glob("*.json")):
            registrations.extend(self._load_project_manifest(manifest, context))
        return registrations

    def _load_project_manifest(self, manifest: Path, context: ToolContext) -> list[ToolRegistration]:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Plugin manifest is not valid JSON: {manifest}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Plugin manifest must contain one object: {manifest}")
        if payload.get("enabled", True) is False:
            return []
        module_name = payload.get("module")
        factory_name = payload.get("factory", "load_tools")
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError(f"Plugin manifest requires non-empty module: {manifest}")
        if not isinstance(factory_name, str) or not factory_name.strip():
            raise ValueError(f"Plugin manifest factory must be non-empty: {manifest}")

        sys.path.insert(0, str(self.repo_root))
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, factory_name)
            return _coerce_registrations(factory(context), str(manifest))
        finally:
            try:
                sys.path.remove(str(self.repo_root))
            except ValueError:
                pass


def _coerce_registrations(value: Any, source: str) -> list[ToolRegistration]:
    if not isinstance(value, list) or not all(isinstance(item, ToolRegistration) for item in value):
        raise ValueError(f"Plugin {source} must return list[ToolRegistration].")
    return list(value)
