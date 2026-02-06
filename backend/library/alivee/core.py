import torch
import torch.nn as nn
import torch.nn.functional as F

class NestedLinear(nn.Module):

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad = False
        self.base.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.base(x)
        if not self.training:
            return base_out
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def consolidate_memory(self, rate=0.01):
        with torch.no_grad():
            delta = self.lora_B @ self.lora_A * self.scaling
            self.base.weight += delta * rate
            self.lora_A.data.normal_(0, 0.02)
            self.lora_B.data.zero_()

class TriStateAttention(nn.Module):

    def __init__(self, d_model, num_heads, rank=4):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = NestedLinear(d_model, d_model, rank=rank)
        self.W_k = NestedLinear(d_model, d_model, rank=rank)
        self.W_v = NestedLinear(d_model, d_model, rank=rank)
        self.W_o = NestedLinear(d_model, d_model, rank=rank)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.W_q(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / self.d_k ** 0.5
        attn = F.softmax(scores, dim=-1)
        context = attn @ v
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_o(context)

    def consolidate_memory(self, rate=0.01):
        self.W_q.consolidate_memory(rate)
        self.W_k.consolidate_memory(rate)
        self.W_v.consolidate_memory(rate)
        self.W_o.consolidate_memory(rate)

class TriStateBrain(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.brain_type == 'transformer':
            assert config.input_dim % config.context_len == 0, 'input_dim must be divisible by context_len for Transformer'
            self.embed_dim = config.input_dim // config.context_len
            self.feature_proj = NestedLinear(self.embed_dim, config.hidden_dim, rank=config.lora_rank)
            layers = []
            for _ in range(config.num_layers):
                layers.append(nn.ModuleDict({'attn': TriStateAttention(config.hidden_dim, config.num_heads, rank=config.lora_rank), 'ffn': nn.Sequential(NestedLinear(config.hidden_dim, config.hidden_dim * 2, rank=config.lora_rank), nn.GELU(), NestedLinear(config.hidden_dim * 2, config.hidden_dim, rank=config.lora_rank)), 'ln1': nn.LayerNorm(config.hidden_dim), 'ln2': nn.LayerNorm(config.hidden_dim)}))
            self.layers = nn.ModuleList(layers)
            self.head = nn.Linear(config.hidden_dim * config.context_len, config.output_dim)
        else:
            self.layers = nn.ModuleList()
            self.layers.append(NestedLinear(config.input_dim, config.hidden_dim, rank=config.lora_rank))
            for _ in range(config.num_layers - 1):
                self.layers.append(NestedLinear(config.hidden_dim, config.hidden_dim, rank=config.lora_rank))
            self.head = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        if self.config.brain_type == 'transformer':
            b, total_dim = x.shape
            x = x.view(b, self.config.context_len, self.embed_dim)
            x = self.feature_proj(x)
            for layer in self.layers:
                residual = x
                x = layer['ln1'](x)
                x = residual + layer['attn'](x)
                residual = x
                x = layer['ln2'](x)
                ffn_in = x
                for module in layer['ffn']:
                    ffn_in = module(ffn_in)
                x = residual + ffn_in
            x = x.view(b, -1)
            return torch.sigmoid(self.head(x))
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x = F.gelu(x)
                if i < len(self.layers) - 1:
                    x = F.dropout(x, p=self.config.dropout, training=self.training)
            return torch.sigmoid(self.head(x))