from __future__ import annotations

from types import SimpleNamespace

import pytest

from my_agent.evaluation.memory_benchmark.api_config import ApiEndpoint
from my_agent.evaluation.memory_benchmark.api_embedding import (
    MemoryBenchmarkApiEmbeddingEncoder,
)
from my_agent.policy.identity import canonical_sha256


def _endpoint() -> ApiEndpoint:
    return ApiEndpoint(
        api_key="secret",
        base_url="https://example.test/v1",
        model="text-embedding-v4",
        endpoint_hash=canonical_sha256("embedding-endpoint"),
    )


class _Embeddings:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _client(*responses: object) -> SimpleNamespace:
    return SimpleNamespace(embeddings=_Embeddings(list(responses)))


def _response(*items: tuple[int, list[float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in items]
    )


def test_embedding_request_and_empty_input() -> None:
    client = _client(_response((0, [1.0, 2.0])))
    encoder = MemoryBenchmarkApiEmbeddingEncoder(_endpoint(), client=client)

    assert encoder.encode_queries(()) == ()
    assert encoder.encode_queries(("query",)) == ((1.0, 2.0),)
    assert client.embeddings.requests == [
        {"model": "text-embedding-v4", "input": ["query"]}
    ]


def test_embedding_batch_is_restored_by_provider_index() -> None:
    client = _client(_response((1, [3.0, 4.0]), (0, [1.0, 2.0])))
    encoder = MemoryBenchmarkApiEmbeddingEncoder(_endpoint(), client=client)

    assert encoder.encode_documents(("first", "second")) == (
        (1.0, 2.0),
        (3.0, 4.0),
    )


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (_response((0, [1.0])), "wrong vector count"),
        (_response((0, []), (1, [])), "must not be empty"),
        (_response((0, [float("nan")]), (1, [1.0])), "must be finite"),
        (_response((0, [float("inf")]), (1, [1.0])), "must be finite"),
        (_response((0, [1.0]), (1, [1.0, 2.0])), "consistent dimension"),
    ],
)
def test_embedding_rejects_invalid_responses(
    response: object, match: str
) -> None:
    encoder = MemoryBenchmarkApiEmbeddingEncoder(_endpoint(), client=_client(response))

    with pytest.raises(ValueError, match=match):
        encoder.encode_documents(("first", "second"))


def test_embedding_rejects_dimension_drift() -> None:
    encoder = MemoryBenchmarkApiEmbeddingEncoder(
        _endpoint(),
        client=_client(_response((0, [1.0, 2.0])), _response((0, [1.0]))),
    )

    encoder.encode_queries(("first",))
    with pytest.raises(ValueError, match="dimension changed"):
        encoder.encode_queries(("second",))


def test_embedding_api_errors_propagate_and_metrics_are_differential() -> None:
    client = _client(_response((0, [1.0, 2.0])), RuntimeError("provider failed"))
    ticks = iter((0.0, 0.5, 1.0, 1.25))
    encoder = MemoryBenchmarkApiEmbeddingEncoder(
        _endpoint(), client=client, clock=lambda: next(ticks)
    )

    encoder.encode_queries(("first",))
    snapshot = encoder.metrics_snapshot()
    with pytest.raises(RuntimeError, match="provider failed"):
        encoder.encode_queries(("second",))
    delta = encoder.metrics_since(snapshot)

    assert snapshot.calls == 1
    assert snapshot.encoded_texts == 1
    assert snapshot.dimension == 2
    assert delta.calls == 1
    assert delta.encoded_texts == 0
    assert delta.elapsed_sec == pytest.approx(0.25)
    assert delta.dimension == 2
