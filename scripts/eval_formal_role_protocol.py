#!/usr/bin/env python3
"""Evaluate selection, action, writing, and maintenance decision protocols."""

from __future__ import annotations

import argparse
import json

from my_agent.evaluation.formal_role_protocol import (
    run_formal_role_protocol_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-events", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = run_formal_role_protocol_evaluation(
        decision_events_path=args.decision_events,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
