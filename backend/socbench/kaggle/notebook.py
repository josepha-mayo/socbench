"""Generate Kaggle notebook for training a dataset."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from socbench.training.trainer import generate_train_script


def dataset_safe_id(dataset_id: str) -> str:
    """Return the Kaggle-safe dataset slug used for dataset IDs."""
    safe_id = re.sub(r"[^a-z0-9]+", "-", dataset_id.lower()).strip("-")
    return safe_id[:50]


def kernel_slug_for(dataset_id: str, prefix: str = "socbench-train-") -> str:
    """Return the short Kaggle kernel slug for a dataset_id."""
    safe_id = dataset_safe_id(dataset_id)
    digest = hashlib.md5(safe_id.encode("utf-8")).hexdigest()[:4]
    short_id = (safe_id[:7] + digest).strip("-").strip("_")
    if not short_id:
        short_id = digest or "data"
    short_id = short_id[:11]
    return f"{prefix}{short_id}"


def generate_notebook(
    dataset_id: str,
    binary_filename: str = "train.bin",
    tokens: int = 1_000_000_000,
    output_dir: str = "/kaggle/working",
    dataset_owner: str | None = None,
    kaggle_dataset_slug: str | None = None,
) -> dict:
    """Generate a Kaggle notebook (ipynb format) for training a dataset.

    Returns the notebook dict ready to be written as .ipynb.
    """
    safe_id = dataset_safe_id(dataset_id)
    kernel_slug = kernel_slug_for(dataset_id)
    title = kernel_slug

    # Kaggle mounts user datasets under /kaggle/input/<dataset-slug>/.
    owner = dataset_owner or "<owner>"
    kernel_owner = dataset_owner or "socbench"
    dataset_slug = kaggle_dataset_slug or safe_id
    dataset_mount_path = f"/kaggle/input/{dataset_slug}"

    # Generate training script content
    train_script = generate_train_script(
        dataset_bin_path=f"{dataset_mount_path}/{binary_filename}",
        output_dir=output_dir,
        tokens=tokens,
    )

    train_script_literal = json.dumps(train_script)

    notebook = {
        "cells": [
            {
                "id": "socbench-title",
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\\n",
                    f"Training GPT-2 124M on `{dataset_id}` for {tokens:,} tokens.\\n",
                ],
            },
            {
                "id": "socbench-setup",
                "cell_type": "code",
                "metadata": {"trusted": True},
                "source": [
                    "# Setup\\n",
                    "import os\\n",
                    "import subprocess\\n",
                    "import sys\\n",
                    "\\n",
                    "# Install dependencies\\n",
                    "subprocess.run([\\n",
                    "    sys.executable, '-m', 'pip', 'install', '-q',\\n",
                    "    'torch', 'numpy', 'tiktoken', 'datasets', 'huggingface_hub'\\n",
                    "])\\n",
                    "\\n",
                    "os.environ['HF_HUB_DISABLE_XET'] = '1'\\n",
                    "os.environ['HF_HOME'] = '/kaggle/working/hf_cache'\\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
            {
                "id": "socbench-verify-data",
                "cell_type": "code",
                "metadata": {"trusted": True},
                "source": [
                    "# Verify data is available\\n",
                    "import os\\n",
                    f"data_path = '{dataset_mount_path}/{binary_filename}'\\n",
                    "if os.path.exists(data_path):\\n",
                    "    size_gb = os.path.getsize(data_path) / (1024**3)\\n",
                    "    print(f'Data found: {size_gb:.2f} GB')\\n",
                    "else:\\n",
                    "    print(f'ERROR: Data not found at {data_path}')\\n",
                    "    print('Available inputs:')\\n",
                    "    for root, dirs, files in os.walk('/kaggle/input/'):\\n",
                    "        for f in files:\\n",
                    "            print(os.path.join(root, f))\\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
            {
                "id": "socbench-train",
                "cell_type": "code",
                "metadata": {"trusted": True},
                "source": [
                    "# Write and run the training script\\n",
                    "import os\\n",
                    "import subprocess\\n",
                    "import sys\\n",
                    "from pathlib import Path\\n",
                    "\\n",
                    "os.environ['NCCL_P2P_DISABLE'] = '1'\\n",
                    "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\\n",
                    "\\n",
                    "train_path = Path('/kaggle/working/train.py')\\n",
                    "train_script = " + train_script_literal + "\\n",
                    "train_path.write_text(train_script)\\n",
                    "\\n",
                    "with open('train.log', 'w') as out, open('train.err', 'w') as err:\\n",
                    "    p = subprocess.run([sys.executable, str(train_path)], stdout=out, stderr=err)\\n",
                    "\\n",
                    "print(f'Training exited with code {p.returncode}')\\n",
                    "if p.returncode != 0:\\n",
                    "    raise RuntimeError('Training failed. See train.log and train.err.')\\n",
                    "\\n",
                    "# Copy compact result artifacts to the Kaggle output root.\\n",
                    "import shutil\\n",
                    f"result_dir = Path('{output_dir}')\\n",
                    "for name in ['loss_curve.json', 'eval_results.json']:\\n",
                    "    src = result_dir / name\\n",
                    "    if src.exists():\\n",
                    "        shutil.copy2(src, Path('/kaggle/working') / name)\\n",
                    "for name in ['train.log', 'train.err']:\\n",
                    "    src = Path(name)\\n",
                    "    if src.exists():\\n",
                    "        shutil.copy2(src, Path('/kaggle/working') / f'socbench_{name}')\\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
        ],
        "metadata": {
            "kaggle": {
                "accelerator": "GPU",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    kernel_metadata = {
        "id": f"{kernel_owner}/{kernel_slug}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
                "dataset_sources": [f"{owner}/{dataset_slug}"],
            }

    return {"notebook": notebook, "kernel_metadata": kernel_metadata, "kernel_slug": kernel_slug}


def save_notebook(output_dir: str, dataset_id: str, **kwargs) -> dict:
    """Save notebook and kernel metadata to disk."""
    result = generate_notebook(dataset_id, **kwargs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(result["notebook"], f, indent=1)

    with open(out / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(result["kernel_metadata"], f, indent=2)

    return {"path": str(out), "slug": result["kernel_slug"]}
