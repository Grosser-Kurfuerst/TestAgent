from __future__ import annotations

import argparse

from my_agent.cli.common import CliContext, positive_top_k, section
from my_agent.repo import RepoIndexer


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    index_parser = subparsers.add_parser("index", help="Preview repository context without calling an LLM.")
    index_parser.add_argument("--repo", required=True, help="Target repository path.")
    index_parser.add_argument("--query", default="", help="Optional retrieval query.")
    index_parser.add_argument("--top-k", type=positive_top_k, default=8, help="Number of retrieved files.")
    index_parser.set_defaults(_handler=handle)

    retrieve_parser = subparsers.add_parser("retrieve", help="Run lightweight lexical retrieval over a repository.")
    retrieve_parser.add_argument("--repo", required=True, help="Target repository path.")
    retrieve_parser.add_argument("--query", required=True, help="Search query.")
    retrieve_parser.add_argument("--top-k", type=positive_top_k, default=5, help="Number of retrieved files.")
    retrieve_parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    _ = ctx
    if args.command == "index":
        snapshot = RepoIndexer(args.repo).snapshot(query=args.query, top_k=args.top_k)
        print(section("Repository tree", snapshot.tree))
        print()
        print(section("Symbol index", snapshot.symbols))
        print()
        print(section("Retrieval notes", snapshot.retrieval_notes))
        print()
        print(section("Project rules", snapshot.project_rules))
        print()
        print(section("Important file previews", snapshot.file_summaries))
        return 0
    if args.command == "retrieve":
        print(RepoIndexer(args.repo).retrieve(query=args.query, top_k=args.top_k))
        return 0
    raise ValueError(f"Unknown repository command: {args.command}")
