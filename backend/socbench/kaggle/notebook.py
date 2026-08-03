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
    output_dir: str = "/kaggle/temp/socbench_output",
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
                    "import json\\n",
                    "import os\\n",
                    "import subprocess\\n",
                    "import sys\\n",
                    "\\n",
                    "# Install dependencies\\n",
                    "subprocess.run([\\n",
                    "    sys.executable, '-m', 'pip', 'install', '-q',\\n",
                    "    'tiktoken', 'datasets', 'huggingface_hub'\\n",
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
                    "os.environ['SOCBENCH_TRAIN_DEVICE'] = 'cpu'\\n",
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
                    "    if Path('train.err').exists():\\n",
                    "        sys.stderr.write(Path('train.err').read_text()[-4000:])\\n",
                    "    raise RuntimeError('Training failed. See train.log and train.err.')\\n",
                    "\\n",
                    "# Copy compact result artifacts to the Kaggle output root.\\n",
                    "import shutil\\n",
                    f"result_dir = Path('{output_dir}')\\n",
                    "summary = {\\n",
                    f"    'dataset_id': {json.dumps(dataset_id)},\\n",
                    f"    'tokens_budget': {tokens},\\n",
                    f"    'output_dir': {json.dumps(output_dir)},\\n",
                    "}\\n",
                    "for name in ['loss_curve.json', 'eval_results.json']:\\n",
                    "    src = result_dir / name\\n",
                    "    if src.exists():\\n",
                    "        shutil.copy2(src, Path('/kaggle/working') / name)\\n",
                    "        summary[name] = json.loads(src.read_text())\\n",
                    "for name in ['train.log', 'train.err']:\\n",
                    "    src = Path(name)\\n",
                    "    if src.exists():\\n",
                    "        shutil.copy2(src, Path('/kaggle/working') / f'socbench_{name}')\\n",
                    "Path('/kaggle/working/socbench_result.json').write_text(json.dumps(summary, indent=2))\\n",
                    "print('SOCBENCH_RESULT_JSON=' + json.dumps(summary), file=sys.stderr)\\n",
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


def generate_kernel_script(
    dataset_id: str,
    binary_filename: str = "train.bin",
    tokens: int = 1_000_000_000,
    output_dir: str = "/kaggle/temp/socbench_output",
    dataset_owner: str | None = None,
    kaggle_dataset_slug: str | None = None,
) -> dict:
    """Generate a plain Python Kaggle script kernel for training."""
    safe_id = dataset_safe_id(dataset_id)
    kernel_slug = kernel_slug_for(dataset_id)
    title = kernel_slug
    owner = dataset_owner or "<owner>"
    kernel_owner = dataset_owner or "socbench"
    dataset_slug = kaggle_dataset_slug or safe_id
    dataset_mount_path = f"/kaggle/input/{dataset_slug}"
    train_script = generate_train_script(
        dataset_bin_path="__SOCBENCH_DATA_PATH__",
        output_dir=output_dir,
        tokens=tokens,
    )
    source = f"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'tiktoken', 'datasets', 'huggingface_hub'
])

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HOME'] = '/kaggle/working/hf_cache'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['SOCBENCH_TRAIN_DEVICE'] = 'cpu'

candidate_paths = [
    Path({json.dumps(f"{dataset_mount_path}/{binary_filename}")}),
    Path({json.dumps(f"/kaggle/input/datasets/{owner}/{dataset_slug}/{binary_filename}")}),
]
data_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
if not data_path.exists():
    print('ERROR: Data not found at ' + str(data_path), file=sys.stderr)
    for root, dirs, files in os.walk('/kaggle/input/'):
        for file_name in files:
            print(os.path.join(root, file_name), file=sys.stderr)
    raise SystemExit(1)

print(f'Data found: {{data_path.stat().st_size / (1024**2):.2f}} MB')
train_path = Path('/kaggle/working/train.py')
train_script = {json.dumps(train_script)}.replace('__SOCBENCH_DATA_PATH__', str(data_path))
train_path.write_text(train_script, encoding='utf-8')

with open('/kaggle/working/socbench_train.log', 'w') as out, open('/kaggle/working/socbench_train.err', 'w') as err:
    proc = subprocess.run([sys.executable, str(train_path)], stdout=out, stderr=err)

if proc.returncode != 0:
    err_path = Path('/kaggle/working/socbench_train.err')
    if err_path.exists():
        print(err_path.read_text(errors='replace')[-4000:], file=sys.stderr)
    raise SystemExit(proc.returncode)

result_dir = Path({json.dumps(output_dir)})
summary = {{
    'dataset_id': {json.dumps(dataset_id)},
    'tokens_budget': {tokens},
    'output_dir': {json.dumps(output_dir)},
}}
for name in ['loss_curve.json', 'eval_results.json']:
    src = result_dir / name
    if src.exists():
        shutil.copy2(src, Path('/kaggle/working') / name)
        summary[name] = json.loads(src.read_text())

Path('/kaggle/working/socbench_result.json').write_text(json.dumps(summary, indent=2))
print('SOCBENCH_RESULT_JSON=' + json.dumps(summary), file=sys.stderr)
"""
    kernel_metadata = {
        "id": f"{kernel_owner}/{kernel_slug}",
        "title": title,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{dataset_slug}"],
    }
    return {"source": source.strip() + "\n", "kernel_metadata": kernel_metadata, "kernel_slug": kernel_slug}


def save_script_kernel(output_dir: str, dataset_id: str, **kwargs) -> dict:
    """Save a plain Python Kaggle script kernel and metadata to disk."""
    result = generate_kernel_script(dataset_id, **kwargs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(result["source"], encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(result["kernel_metadata"], indent=2) + "\n",
        encoding="utf-8",
    )
    return {"path": str(out), "slug": result["kernel_slug"]}


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
