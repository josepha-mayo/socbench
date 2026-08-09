"""GPT-2 124M training config."""

from __future__ import annotations

from dataclasses import dataclass


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
    """Training hyperparameters for GPT-2 124M."""

    # Optimizer
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Schedule
    warmup_tokens: int = 10_000_000
    lr_decay_to: float = 3e-5

    # Batch
    batch_size: int = 8  # Per GPU
    gradient_accumulation_steps: int = 64
    max_iters: int = 0  # Set dynamically based on token budget

    # Token budget
    tokens_per_dataset: int = 1_000_000_000  # 1B tokens
    max_dataset_repeats: float = 8.0

    # Fail-closed calibration and divergence gates
    calibration_iters: int = 100
    min_calibration_improvement: float = 0.005
    max_val_loss_increase: float = 0.05
    divergence_patience: int = 2

    # Precision
    dtype: str = "float16"  # T4 has no BF16
    use_flash_attention: bool = True

    # Checkpointing
    eval_interval: int = 500
    log_interval: int = 100
    save_interval: int = 2000
    eval_iters: int = 200

    # Hardware
    compile: bool = False
    ddp: bool = True  # DistributedDataParallel


MODEL = ModelConfig()
TRAIN = TrainConfig()
