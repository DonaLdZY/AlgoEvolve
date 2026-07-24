from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.memory.global_memory import GlobalMemoryLayer


def _node(node_id: str = "node-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=node_id,
        stage="draft",
        plan="try a solver",
        code="def predict(model_path, data):\n    return data\n",
        code_summary="solver",
        metric=SimpleNamespace(value=1.0, maximize=False),
        is_buggy=False,
        exec_time=1.0,
    )


def test_placeholder_remote_embedding_url_fails_before_provider_initialization(tmp_path) -> None:
    with pytest.raises(ValueError, match="unresolved placeholder"):
        GlobalMemoryLayer(
            memory_dir=str(tmp_path),
            embedding_backend="openai",
            embedding_api_key="secret",
            embedding_base_url="https://{WorkspaceId}.example.com/v1",
            embedding_model="embedding-model",
        )


def test_runtime_embedding_failure_disables_memory_and_rolls_back_record(tmp_path) -> None:
    class FailingRetriever:
        vector_index = None
        calls = 0

        def build_index(self, _records, _texts):
            self.calls += 1
            raise ConnectionError("provider unavailable")

    layer = GlobalMemoryLayer.__new__(GlobalMemoryLayer)
    layer.memory_dir = tmp_path
    layer.similarity_threshold = 0.7
    layer.disabled_reason = None
    layer.records = []
    layer.node_metadata_map = {}
    layer.retriever = FailingRetriever()

    assert layer.save_node(_node()) is False
    assert layer.disabled_reason is not None
    assert layer.records == []
    assert layer.node_metadata_map == {}
    assert layer.retriever.calls == 1

    assert layer.save_node(_node("node-2")) is False
    assert layer.retriever.calls == 1
