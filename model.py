"""GPT with persistent parameter tensors and an explicit functional forward.

Read GPT.__init__ for parameter shapes, initialize_parameters for initial values,
and GPT.forward for the complete computation. No trainable layers are created
inside forward. Parameters are registered once in self.params.
"""

import inspect
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # INIT | TABLE1: baseline σ/σ² and constant Norm/bias values, not paper scaling.
    init_std: float = 0.02
    attn_out_init_std: float | None = None
    mlp_out_init_std: float | None = None
    norm_weight_init: float = 1.0
    norm_bias_init: float = 0.0
    linear_bias_init: float = 0.0

    def __post_init__(self):
        # Resolve defaults here, before GPT is constructed. Explicit values win.
        # NanoGPT default: output projection entries ~ N(0, σ²/(2L)).
        default_std = self.init_std / math.sqrt(2 * self.n_layer)
        if self.attn_out_init_std is None:
            self.attn_out_init_std = default_std
        if self.mlp_out_init_std is None:
            self.mlp_out_init_std = default_std


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        N = config.n_embd
        self.params = nn.ParameterDict()
        p = self.params

        # Registration helpers only create/store tensors; they do no forward work.
        def add_norm(name):
            p[name + '_weight'] = nn.Parameter(torch.ones(N))
            p[name + '_bias'] = nn.Parameter(torch.zeros(N)) if config.bias else None

        def add_linear(name, fan_in, fan_out, bias=True):
            # Temporary Linear supplies the original constructor initialization.
            # Only its parameters survive; the module is never used for computation.
            layer = nn.Linear(fan_in, fan_out, bias=bias)
            p[name + '_weight'] = layer.weight
            p[name + '_bias'] = layer.bias

        # Preserve NanoGPT's parameter creation order and random draws.
        p['token_embedding'] = nn.Embedding(config.vocab_size, N).weight
        p['position_embedding'] = nn.Embedding(config.block_size, N).weight
        for i in range(config.n_layer):
            add_norm(f'h{i}_attn_norm')
            add_linear(f'h{i}_qkv', N, 3 * N, config.bias)
            add_linear(f'h{i}_attn_out', N, N, config.bias)
            add_norm(f'h{i}_mlp_norm')
            add_linear(f'h{i}_mlp_in', N, 4 * N, config.bias)
            add_linear(f'h{i}_mlp_out', 4 * N, N, config.bias)
        add_norm('final_norm')
        add_linear('head', N, config.vocab_size, bias=False)

        # One shared parameter receives both embedding and output-head gradients.
        p['token_embedding'] = p['head_weight']
        self.initialize_parameters()
        print('number of parameters: %.2fM' % (self.get_num_params() / 1e6,))

    def initialize_parameters(self):
        """INIT | TABLE1: explicit matrix standard deviations and Norm γ/β.

        Called by __init__, never by forward. Settings below are initial values;
        the optimizer can subsequently update these same parameter tensors.
        Keep random draws in NanoGPT order, including the tied head and the
        second initialization of residual projections, for exact baseline parity.
        """
        cfg, p = self.config, self.params
        sigma = cfg.init_std  # Standard deviation σ; variance is σ².

        # 1. Token and learned position embeddings: each entry ~ N(0, σ²).
        nn.init.normal_(p['token_embedding'], mean=0.0, std=sigma)
        nn.init.normal_(p['position_embedding'], mean=0.0, std=sigma)

        for i in range(cfg.n_layer):
            # 2. Attention Pre-Norm γ; all optional biases are filled below.
            nn.init.constant_(p[f'h{i}_attn_norm_weight'], cfg.norm_weight_init)

            # 3. Attention: packed W_Q/W_K/W_V, then output projection W_O.
            # All entries start ~ N(0, σ²); step 7 sets W_O's final initial value.
            nn.init.normal_(p[f'h{i}_qkv_weight'], mean=0.0, std=sigma)
            nn.init.normal_(p[f'h{i}_attn_out_weight'], mean=0.0, std=sigma)

            # 4. MLP Pre-Norm: its own γ, with the same initial value.
            nn.init.constant_(p[f'h{i}_mlp_norm_weight'], cfg.norm_weight_init)

            # 5. MLP: expansion and output weights start ~ N(0, σ²).
            # Step 7 sets the output projection's final initial value.
            nn.init.normal_(p[f'h{i}_mlp_in_weight'], mean=0.0, std=sigma)
            nn.init.normal_(p[f'h{i}_mlp_out_weight'], mean=0.0, std=sigma)

        # 6. Final LayerNorm and vocabulary head (no head bias).
        nn.init.constant_(p['final_norm_weight'], cfg.norm_weight_init)
        # This also overwrites token_embedding: they share the SAME tensor.
        nn.init.normal_(p['head_weight'], mean=0.0, std=sigma)

        # 7. Output projections: use the standard deviations supplied by config.
        # This changes initial weights, not the residual addition in forward.
        for i in range(cfg.n_layer):
            nn.init.normal_(p[f'h{i}_attn_out_weight'], mean=0.0, std=cfg.attn_out_init_std)
            nn.init.normal_(p[f'h{i}_mlp_out_weight'], mean=0.0, std=cfg.mlp_out_init_std)

        # 8. Optional biases: LayerNorm β and linear b have separate initial values.
        # Constant fills consume no random numbers, so they can be grouped here.
        if cfg.bias:
            for i in range(cfg.n_layer):
                nn.init.constant_(p[f'h{i}_attn_norm_bias'], cfg.norm_bias_init)
                nn.init.constant_(p[f'h{i}_mlp_norm_bias'], cfg.norm_bias_init)
                nn.init.constant_(p[f'h{i}_qkv_bias'], cfg.linear_bias_init)
                nn.init.constant_(p[f'h{i}_attn_out_bias'], cfg.linear_bias_init)
                nn.init.constant_(p[f'h{i}_mlp_in_bias'], cfg.linear_bias_init)
                nn.init.constant_(p[f'h{i}_mlp_out_bias'], cfg.linear_bias_init)
            nn.init.constant_(p['final_norm_bias'], cfg.norm_bias_init)

    def get_num_params(self, non_embedding=True):
        """Count unique parameters; optionally exclude positions as in NanoGPT."""
        count = sum(p.numel() for p in self.parameters())
        return count - self.params['position_embedding'].numel() if non_embedding else count

    def forward(self, idx, targets=None):
        """Token IDs → embeddings → repeated attention/MLP → final norm → logits."""
        cfg, p = self.config, self.params
        B, T = idx.shape
        N, H = cfg.n_embd, cfg.n_head
        D = N // H
        assert T <= cfg.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)

        # 1. Embedding lookup and learned absolute positions.
        token_embedding = F.embedding(idx, p['token_embedding'])
        position_embedding = F.embedding(pos, p['position_embedding'])
        x = F.dropout(token_embedding + position_embedding, cfg.dropout, self.training)

        for i in range(cfg.n_layer):
            # 2. Pre-Norm: explicit γ and optional β for this layer.
            u = F.layer_norm(x, (N,), p[f'h{i}_attn_norm_weight'],
                             p[f'h{i}_attn_norm_bias'], eps=1e-5)

            # 3. Project Q/K/V, split heads, and apply causal attention.
            q, k, v = F.linear(u, p[f'h{i}_qkv_weight'], p[f'h{i}_qkv_bias']).split(N, dim=-1)
            k = k.view(B, T, H, D).transpose(1, 2)
            q = q.view(B, T, H, D).transpose(1, 2)
            v = v.view(B, T, H, D).transpose(1, 2)
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                dropout_p=cfg.dropout if self.training else 0.0,
            )  # Default QK score multiplier: 1 / √D.
            a = a.transpose(1, 2).contiguous().view(B, T, N)
            a = F.linear(a, p[f'h{i}_attn_out_weight'], p[f'h{i}_attn_out_bias'])
            a = F.dropout(a, cfg.dropout, self.training)
            x = x + a

            # 4. Pre-Norm → linear → GELU → linear → residual addition.
            u = F.layer_norm(x, (N,), p[f'h{i}_mlp_norm_weight'],
                             p[f'h{i}_mlp_norm_bias'], eps=1e-5)
            f = F.linear(u, p[f'h{i}_mlp_in_weight'], p[f'h{i}_mlp_in_bias'])
            f = F.gelu(f)
            f = F.linear(f, p[f'h{i}_mlp_out_weight'], p[f'h{i}_mlp_out_bias'])
            f = F.dropout(f, cfg.dropout, self.training)
            x = x + f

        # 5. Final norm and the shared vocabulary projection.
        x = F.layer_norm(x, (N,), p['final_norm_weight'], p['final_norm_bias'], eps=1e-5)
        if targets is None:
            return F.linear(x[:, [-1], :], p['head_weight']), None
        logits = F.linear(x, p['head_weight'])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             adam_eps=1e-8):
        """List existing parameters once, assign update settings, and return AdamW."""
        # 1. List the same trainable tensors used by forward, in registration order.
        # The output head shares token_embedding: list that tensor only once.
        p = self.params
        parameter_roles = [
            (p['token_embedding'], 'embedding'),
            (p['position_embedding'], 'embedding'),
        ]
        for i in range(self.config.n_layer):
            parameter_roles.extend([
                (p[f'h{i}_attn_norm_weight'], 'hidden_norm'),
                (p[f'h{i}_attn_norm_bias'],   'hidden_norm'),
                (p[f'h{i}_qkv_weight'],      'hidden_weight'),
                (p[f'h{i}_qkv_bias'],        'hidden_bias'),
                (p[f'h{i}_attn_out_weight'], 'hidden_weight'),
                (p[f'h{i}_attn_out_bias'],   'hidden_bias'),
                (p[f'h{i}_mlp_norm_weight'], 'hidden_norm'),
                (p[f'h{i}_mlp_norm_bias'],   'hidden_norm'),
                (p[f'h{i}_mlp_in_weight'],   'hidden_weight'),
                (p[f'h{i}_mlp_in_bias'],     'hidden_bias'),
                (p[f'h{i}_mlp_out_weight'],  'hidden_weight'),
                (p[f'h{i}_mlp_out_bias'],    'hidden_bias'),
            ])
        parameter_roles.extend([
            (p['final_norm_weight'], 'final_norm'),
            (p['final_norm_bias'],   'final_norm'),
        ])

        # 2. NanoGPT updates all of these parameters, using two decay settings.
        # Embeddings and hidden matrices decay; all Norm parameters and biases do not.
        group_for_role = {
            'embedding': 'decay',
            'hidden_norm': 'no_decay',
            'hidden_weight': 'decay',
            'hidden_bias': 'no_decay',
            'final_norm': 'no_decay',
        }
        update_settings = {
            'decay': {'weight_decay': weight_decay},
            'no_decay': {'weight_decay': 0.0},
        }

        # Add each existing tensor to its selected group, preserving parameter order.
        # Missing biases are None; frozen parameters must not enter the optimizer.
        grouped_parameters = {group: [] for group in group_for_role.values()}
        for parameter, role in parameter_roles:
            if parameter is not None and parameter.requires_grad:
                group = group_for_role[role]
                grouped_parameters[group].append(parameter)
        optim_groups = [
            {'params': params, **update_settings[group]}
            for group, params in grouped_parameters.items()
        ]

        # 3. Return one AdamW object holding references to these model parameters.
        # This call does not update weights. loss.backward() fills their .grad;
        # optimizer.step() later updates the same tensors that forward reads.
        # lr/betas/eps below are defaults; settings written in a group override them.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            eps=adam_eps,
            **extra_args,
        )

        # Parameter counts by dimension, as in NanoGPT's original diagnostics.
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        matrix_params = [parameter for parameter in parameters if parameter.dim() >= 2]
        vector_params = [parameter for parameter in parameters if parameter.dim() < 2]
        print(f"num decayed parameter tensors: {len(matrix_params)}, "
              f"with {sum(p.numel() for p in matrix_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(vector_params)}, "
              f"with {sum(p.numel() for p in vector_params):,} parameters")
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
