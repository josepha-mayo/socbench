"""Generate Kaggle notebook for training a dataset."""

from __future__ import annotations

import json
from pathlib import Path

from socbench.training.trainer import generate_train_script


def generate_notebook(
    dataset_id: str,
    binary_filename: str = "train.bin",
    tokens: int = 1_000_000_000,
    output_dir: str = "/kaggle/working",
) -> dict:
    """Generate a Kaggle notebook (ipynb format) for training.

    Returns the notebook dict ready to be written as .ipynb.
    """
    safe_id = dataset_id.replace("/", "_").replace("-", "_")
    kernel_slug = f"socbench-train-{safe_id}"

    # Generate training script content
    train_script = generate_train_script(
        dataset_bin_path=f"/kaggle/input/{safe_id}/{binary_filename}",
        output_dir=output_dir,
        tokens=tokens,
    )

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Socbench Training: {dataset_id}\n",
                    f"Training GPT-2 124M on `{dataset_id}` for {tokens:,} tokens.",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {
                    "trusted": True,
                },
                "source": [
                    "# Setup\n",
                    "import os\n",
                    "os.environ['HF_HUB_DISABLE_XET'] = '1'\n",
                    "os.environ['HF_HOME'] = '/kaggle/working/hf_cache'\n",
                    "!pip install -U torch --quiet\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {
                    "trusted": True,
                },
                "source": [
                    "# Verify data is available\n",
                    "import os\n",
                    f"data_path = '/kaggle/input/{safe_id}/{binary_filename}'\n",
                    "if os.path.exists(data_path):\n",
                    f"    size_gb = os.path.getsize(data_path) / (1024**3)\n",
                    "    print(f'Data found: {size_gb:.2f} GB')\n",
                    "else:\n",
                    "    print(f'ERROR: Data not found at {data_path}')\n",
                    "    print('Available inputs:')\n",
                    "    for root, dirs, files in os.walk('/kaggle/input/'):\n",
                    "        for f in files:\n",
                    "            print(os.path.join(root, f))\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {
                    "trusted": True,
                },
                "source": train_script.split("\n"),
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {
                    "trusted": True,
                },
                "source": [
                    "# Upload results to HuggingFace Hub\n",
                    "# Uncomment and set your token:\n",
                    "# from huggingface_hub import HfApi\n",
                    "# api = HfApi()\n",
                    "# api.upload_folder(\n",
                    "#     folder_path='/kaggle/working',\n",
                    "#     repo_id='your-org/socbench-results',\n",
                    "#     path_in_repo='results/" + safe_id + "',\n",
                    "#     token='hf_your_token',\n",
                    "# )\n",
                    "# print('Results uploaded!')\n",
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
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # Kernel metadata
    kernel_metadata = {
        "id": f"socbench/{kernel_slug}",
        "title": f"Socbench: {dataset_id}",
        "code_file": "",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"socbench/{safe_id}"],
    }

    return {"notebook": notebook, "kernel_metadata": kernel_metadata, "kernel_slug": kernel_slug}


def save_notebook(output_dir: str, dataset_id: str, **kwargs) -> dict:
    """Save notebook and kernel metadata to disk."""
    result = generate_notebook(dataset_id, **kwargs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "notebook.ipynb", "w") as f:
        json.dump(result["notebook"], f, indent=1)

    with open(out / "kernel-metadata.json", "w") as f:
        json.dump(result["kernel_metadata"], f, indent=2)

    return {"path": str(out), "slug": result["kernel_slug"]}
