"""Stable SQLite path and DDL for the Q=30 maintenance ledger."""

EVOLVER_STATE_FILENAME = "evolver_state.sqlite3"

TASK_COMPLETION_DDL = """
CREATE TABLE IF NOT EXISTS task_completion (
  stream_id TEXT NOT NULL,
  memory_project_key TEXT NOT NULL,
  task_id TEXT NOT NULL,
  task_ordinal INTEGER NOT NULL,
  outcome_finalized INTEGER NOT NULL,
  writer_terminal_status TEXT NOT NULL,
  repository_revision_after_writer TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (stream_id, memory_project_key, task_id),
  UNIQUE (stream_id, memory_project_key, task_ordinal)
)
""".strip()

MAINTENANCE_CADENCE_DDL = """
CREATE TABLE IF NOT EXISTS maintenance_cadence (
  stream_id TEXT NOT NULL,
  memory_project_key TEXT NOT NULL,
  cadence_index INTEGER NOT NULL,
  cadence_id TEXT NOT NULL UNIQUE,
  boundary_ordinal INTEGER NOT NULL,
  status TEXT NOT NULL,
  maintenance_plan_id TEXT,
  repository_revision_after TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (stream_id, memory_project_key, cadence_index)
)
""".strip()

TASK_OUTCOME_EVIDENCE_DDL = """
CREATE TABLE IF NOT EXISTS task_outcome_evidence (
  stream_id TEXT NOT NULL,
  memory_project_key TEXT NOT NULL,
  task_id TEXT NOT NULL,
  task_ordinal INTEGER NOT NULL,
  task_group TEXT NOT NULL,
  outcome_json TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (stream_id, memory_project_key, task_id),
  UNIQUE (stream_id, memory_project_key, task_ordinal)
)
""".strip()

LEDGER_DDL = (
    TASK_COMPLETION_DDL,
    MAINTENANCE_CADENCE_DDL,
    TASK_OUTCOME_EVIDENCE_DDL,
)

__all__ = [
    "EVOLVER_STATE_FILENAME",
    "LEDGER_DDL",
    "MAINTENANCE_CADENCE_DDL",
    "TASK_COMPLETION_DDL",
    "TASK_OUTCOME_EVIDENCE_DDL",
]
