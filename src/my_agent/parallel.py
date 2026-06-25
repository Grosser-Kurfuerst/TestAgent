from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def create_bounded_executor(*, max_workers: int, thread_name_prefix: str) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)


def shutdown_executor(executor: ThreadPoolExecutor) -> None:
    executor.shutdown(wait=False, cancel_futures=True)
