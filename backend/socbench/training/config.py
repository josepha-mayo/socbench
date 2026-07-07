"""GPT-2 124M training config optimized for Kaggle 2x T4.

Architecture: Standard GPT-2 (12 layers, 12 heads, d_model=768)
Training: FP16, DDP, NanoGPT format
Budget: 1B tokens per dataset in 12 hours
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """GPT-2 124M architecture config."""

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50_257  # GPT-2 BPE
    dropout: float = 0.0
    bias: bool = True

    @property
    def n_params(self) -> int:
        # Rough estimate
        embed = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        attn = self.n_layer * (4 * self.n_embd * self.n_embd + 4 * self.n_embd)
        ffn = self.n_layer * (2 * 4 * self.n_embd * self.n_embd + 5 * self.n_embd)
        ln = self.n_layer * 4 * self.n_embd
        return embed + attn + ffn + ln


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters for Kaggle 2x T4."""

    # Optimizer
    learning_rate: float = 6e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Schedule
    warmup_tokens: int = 375_000_000  # 375M tokens
    lr_decay_to: float = 6e-5  # Cosine decay to 10% of peak

    # Batch
    batch_size: int = 16  # Per GPU
    gradient_accumulation_steps: int = 32  # Effective batch = 16 * 32 * 2 GPUs = 1024 seqs
    max_iters: int = 0  # Set dynamically based on token budget

    # Token budget
    tokens_per_dataset: int = 1_000_000_000  # 1B tokens

    # Precision
    dtype: str = "float16"  # T4 has no BF16
    use_flash_attention: bool = True

    # Checkpointing
    eval_interval: int = 500
    log_interval: int = 100
    save_interval: int = 2000
    eval_iters: int = 200

    # Hardware
    compile: bool = True  # torch.compile for speed
    ddp: bool = True  # DistributedDataParallel


@dataclass(frozen=True)
class KaggleConfig:
    """Kaggle notebook environment config."""

    gpu_count: int = 2
    gpu_type: str = "T4"
    max_runtime_hours: int = 12
    output_dir: str = "/kaggle/working"
    input_dir: str = "/kaggle/input"
    temp_dir: str = "/kaggle/temp"
    hf_cache: str = "/kaggle/working/hf_cache"

    # Environment
    disable_xet: bool = True  # HF Xet transfer issues on Kaggle
    fp16: bool = True  # T4 only supports FP16, not BF16


# Default configs
MODEL = ModelConfig()
TRAIN = TrainConfig()
KAGGLE = KaggleConfig()
