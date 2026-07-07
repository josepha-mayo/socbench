"""Data preparation — HuggingFace dataset to NanoGPT binary format.

Converts HF datasets to tokenized binary files for efficient training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx

VIEWER_API = "https://datasets-server.huggingface.co"


async def prepare_dataset_binary(
    dataset_id: str,
    output_path: str,
    text_key: str = "text",
    tokenizer_name: str = "gpt2",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> dict:
    """Download and tokenize a dataset into NanoGPT binary format.

    Returns metadata about the prepared data.
    """
    try:
        import tiktoken
        import numpy as np
    except ImportError as e:
        raise ImportError(f"Required packages: {e}") from e

    enc = tiktoken.get_encoding(tokenizer_name)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_tokens: list[int] = []
    total_rows = 0

    async with httpx.AsyncClient(timeout=60) as client:
        # Get dataset info
        try:
            resp = await client.get(
                f"{VIEWER_API}/splits",
                params={"dataset": dataset_id},
                headers=headers,
            )
            resp.raise_for_status()
            splits = resp.json().get("splits", [])
        except Exception as e:
            return {"error": f"Could not fetch splits: {e}", "tokens": 0}

        for split_info in splits[:1]:  # Use first split
            split_name = split_info.get("split", "train")
            config = split_info.get("config", "default")
            num_rows = split_info.get("num_rows", 0)

            if max_samples:
                num_rows = min(num_rows, max_samples)

            # Fetch rows in batches
            offset = 0
            batch_size = 100
            while offset < num_rows:
                try:
                    resp = await client.get(
                        f"{VIEWER_API}/rows",
                        params={
                            "dataset": dataset_id,
                            "config": config,
                            "split": split_name,
                            "offset": offset,
                            "length": batch_size,
                        },
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        break
                    rows = resp.json().get("rows", [])
                    if not rows:
                        break

                    for row in rows:
                        row_data = row.get("row", {})
                        text = ""
                        for key in [text_key, "text", "content", "document", "problem"]:
                            if key in row_data and isinstance(row_data[key], str):
                                text = row_data[key]
                                break

                        if text.strip():
                            tokens = enc.encode_ordinary(text)
                            tokens.append(enc.eot_token)  # End of text token
                            all_tokens.extend(tokens)
                            total_rows += 1

                    offset += len(rows)
                except Exception:
                    break

    if not all_tokens:
        return {"error": "No tokens generated", "tokens": 0}

    # Save as numpy binary (NanoGPT format)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(str(output / "train.bin"))

    # Save metadata
    metadata = {
        "dataset_id": dataset_id,
        "text_key": text_key,
        "tokenizer": tokenizer_name,
        "total_tokens": len(all_tokens),
        "total_rows": total_rows,
        "vocab_size": enc.n_vocab,
        "file_path": str(output / "train.bin"),
        "file_size_bytes": os.path.getsize(str(output / "train.bin")),
    }

    import json

    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
