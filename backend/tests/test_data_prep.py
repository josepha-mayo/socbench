"""Tests for Hugging Face row preparation into NanoGPT binaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from socbench.training.data_prep import extract_training_text, prepare_dataset_binary


def test_extract_training_text_handles_chat_rows():
    row = {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "content": "4"},
        ],
    }

    assert extract_training_text(row) == "Solve 2+2.\n4"


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("fake http error")


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, params=None, headers=None):
        if url.endswith("/splits"):
            return _FakeResponse(
                {
                    "splits": [
                        {
                            "config": "default",
                            "split": "train",
                            "num_rows": 2,
                        }
                    ]
                }
            )
        if url.endswith("/rows"):
            return _FakeResponse(
                {
                    "rows": [
                        {"row": {"prompt": "Say hi", "completion": "Hello"}},
                        {"row": {"messages": [{"content": "Nested text"}]}},
                    ]
                }
            )
        return _FakeResponse({}, status_code=404)


@pytest.mark.asyncio
async def test_prepare_dataset_binary_creates_output_dir_and_uses_fallback_text(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("socbench.training.data_prep.httpx.AsyncClient", _FakeAsyncClient)

    output_dir = tmp_path / "prepared" / "dataset"
    result = await prepare_dataset_binary("fake/dataset", str(output_dir), max_samples=2)

    assert result["total_tokens"] > 0
    assert result["total_rows"] == 2
    assert (output_dir / "train.bin").exists()
    assert (output_dir / "metadata.json").exists()
