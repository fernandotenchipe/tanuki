from dataclasses import dataclass

@dataclass
class AgentConfig:
    input_dim: int = 0
    output_dim: int = 0
    hidden_dim: int = 64
    inner_lr: float = 0.01
    outer_lr: float = 0.005
    lora_rank: int = 4
    dropout: float = 0.1
    fused_bias: bool = True
    brain_type: str = 'mlp'
    num_layers: int = 2
    num_heads: int = 4
    context_len: int = 1