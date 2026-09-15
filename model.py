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
    ### Begin normalization-ablation code ###
    # Branch: codex/normalization-ablation. Type and placement are independent.
    norm_type: str = 'layernorm'  # 'layernorm' or 'rmsnorm', including final norm.
    extra_output_norm: bool = False  # Add Norm before both W_O and W_2.
    norm_eps: float = 1e-5
    ### End normalization-ablation code ###
    # Initial matrix standard deviations and LayerNorm γ/β.
    # Width/depth formulas are evaluated by the caller, not by the GPT model.
    init_std: float = 0.02
    attn_out_init_std: float | None = None
    mlp_out_init_std: float | None = None
    ### Begin CompleteP code ###
    # CompleteP/μP overrides for QKV and MLP expansion initialization.
    # NanoGPT: None keeps init_std; μP/CompleteP: caller passes σ/√width_multiplier.
    qkv_init_std: float | None = None
    mlp_in_init_std: float | None = None
    ### End CompleteP code ###
    norm_weight_init: float = 1.0
    norm_bias_init: float = 0.0
    linear_bias_init: float = 0.0
    ### Begin CompleteP code ###
    # CompleteP forward scales; defaults retain NanoGPT computation.
    # Attention: None -> 1/√D, μP -> 1/D; D is the width of one attention head.
    attention_scale: float | None = None
    input_multiplier: float = 1.0  # Applied after token + position embedding/dropout.
    # Caller passes depth_multiplier**(-alpha); these are NOT weight init scales.
    attn_residual_multiplier: float = 1.0
    mlp_residual_multiplier: float = 1.0
    # Author-compatible: output_alpha / width_multiplier, only with targets.
    training_output_multiplier: float = 1.0
    ### End CompleteP code ###

    def __post_init__(self):
        ### Begin normalization-ablation code ###
        if self.norm_type not in ('layernorm', 'rmsnorm'):
            raise ValueError(f'Unknown norm_type: {self.norm_type}')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('norm_eps must be finite and positive')
        ### End normalization-ablation code ###
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
        ### Begin normalization-ablation code ###
        def add_norm(name, width=N):
            p[name + '_weight'] = nn.Parameter(torch.ones(width))
            # RMSNorm has no beta; config.bias still controls Linear biases.
            use_bias = config.bias and config.norm_type == 'layernorm'
            p[name + '_bias'] = nn.Parameter(torch.zeros(width)) if use_bias else None
        ### End normalization-ablation code ###

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
            ### Begin normalization-ablation code ###
            if config.extra_output_norm:
                # Constant constructors preserve the shared matrices' RNG draws.
                add_norm(f'h{i}_attn_out_norm', N)
                add_norm(f'h{i}_mlp_out_norm', 4 * N)
            ### End normalization-ablation code ###
        add_norm('final_norm')
        add_linear('head', N, config.vocab_size, bias=False)

        # One shared parameter receives both embedding and output-head gradients.
        p['token_embedding'] = p['head_weight']
        self.initialize_parameters()
        print('number of parameters: %.2fM' % (self.get_num_params() / 1e6,))

    def initialize_parameters(self):
        """Initialize embeddings, each layer, then final norm; never called by forward.

        Each shared tensor is initialized once here. Distributions match the
        supplied settings, but draw order differs from the upstream two-pass init.
        """
        cfg, p = self.config, self.params
        sigma = cfg.init_std  # Standard deviation σ; variance is σ².

        ### Begin normalization-ablation code ###
        def init_norm(name):
            nn.init.constant_(p[name + '_weight'], cfg.norm_weight_init)
            if p[name + '_bias'] is not None:
                nn.init.constant_(p[name + '_bias'], cfg.norm_bias_init)
        ### End normalization-ablation code ###

        # NanoGPT embeddings: N(0, σ²). The output head shares token_embedding.
        nn.init.normal_(p['token_embedding'], mean=0.0, std=sigma)
        nn.init.normal_(p['position_embedding'], mean=0.0, std=sigma)

        for i in range(cfg.n_layer):
            ### Begin normalization-ablation code ###
            init_norm(f'h{i}_attn_norm')
            init_norm(f'h{i}_mlp_norm')
            if cfg.extra_output_norm:
                init_norm(f'h{i}_attn_out_norm')
                init_norm(f'h{i}_mlp_out_norm')
            ### End normalization-ablation code ###

            ### Begin CompleteP code ###
            # CompleteP/μP supplies the width-scaled QKV std; NanoGPT uses σ.
            if cfg.qkv_init_std is not None:
                nn.init.normal_(p[f'h{i}_qkv_weight'], mean=0.0, std=cfg.qkv_init_std)
            else:
                nn.init.normal_(p[f'h{i}_qkv_weight'], mean=0.0, std=sigma)
            ### End CompleteP code ###

            # W_O: NanoGPT defaults to σ/√(2L); CompleteP supplies its own std.
            nn.init.normal_(p[f'h{i}_attn_out_weight'], mean=0.0, std=cfg.attn_out_init_std)

            ### Begin CompleteP code ###
            # CompleteP/μP supplies the MLP expansion std; NanoGPT uses σ.
            if cfg.mlp_in_init_std is not None:
                nn.init.normal_(p[f'h{i}_mlp_in_weight'], mean=0.0, std=cfg.mlp_in_init_std)
            else:
                nn.init.normal_(p[f'h{i}_mlp_in_weight'], mean=0.0, std=sigma)
            ### End CompleteP code ###

            # MLP output: same default projection rule, independently configurable.
            nn.init.normal_(p[f'h{i}_mlp_out_weight'], mean=0.0, std=cfg.mlp_out_init_std)

            ### Begin normalization-ablation code ###
            # Norm beta is initialized above; Linear biases remain independent.
            if cfg.bias:
                nn.init.constant_(p[f'h{i}_qkv_bias'], cfg.linear_bias_init)
                nn.init.constant_(p[f'h{i}_attn_out_bias'], cfg.linear_bias_init)
                nn.init.constant_(p[f'h{i}_mlp_in_bias'], cfg.linear_bias_init)
                nn.init.constant_(p[f'h{i}_mlp_out_bias'], cfg.linear_bias_init)
            ### End normalization-ablation code ###

        ### Begin normalization-ablation code ###
        init_norm('final_norm')
        ### End normalization-ablation code ###

    def get_num_params(self, non_embedding=True):
        """Count unique parameters; optionally exclude positions as in NanoGPT."""
        count = sum(p.numel() for p in self.parameters())
        return count - self.params['position_embedding'].numel() if non_embedding else count

    ### Begin normalization-ablation code ###
    def _norm(self, x, name):
        """Normalize the last feature dimension using registered parameters."""
        cfg, p = self.config, self.params
        weight = p[name + '_weight']
        if cfg.norm_type == 'layernorm':
            return F.layer_norm(x, weight.shape, weight, p[name + '_bias'],
                                eps=cfg.norm_eps)
        return F.rms_norm(x, weight.shape, weight, eps=cfg.norm_eps)
    ### End normalization-ablation code ###

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
        ### Begin CompleteP code ###
        # CompleteP input scaling; NanoGPT uses 1.
        x = x * cfg.input_multiplier
        ### End CompleteP code ###

        for i in range(cfg.n_layer):
            # 2. Pre-Norm: explicit γ and optional β for this layer.
            ### Begin normalization-ablation code ###
            u = self._norm(x, f'h{i}_attn_norm')
            ### End normalization-ablation code ###

            # 3. Project Q/K/V, split heads, and apply causal attention.
            q, k, v = F.linear(u, p[f'h{i}_qkv_weight'], p[f'h{i}_qkv_bias']).split(N, dim=-1)
            k = k.view(B, T, H, D).transpose(1, 2)
            q = q.view(B, T, H, D).transpose(1, 2)
            v = v.view(B, T, H, D).transpose(1, 2)
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                dropout_p=cfg.dropout if self.training else 0.0,
                ### Begin CompleteP code ###
                # CompleteP/μP: 1/D; None retains NanoGPT's 1/√D before softmax.
                scale=cfg.attention_scale,
                ### End CompleteP code ###
            )
            a = a.transpose(1, 2).contiguous().view(B, T, N)
            ### Begin normalization-ablation code ###
            # W_O input: normalize across all N concatenated head features.
            if cfg.extra_output_norm:
                a = self._norm(a, f'h{i}_attn_out_norm')
            ### End normalization-ablation code ###
            a = F.linear(a, p[f'h{i}_attn_out_weight'], p[f'h{i}_attn_out_bias'])
            a = F.dropout(a, cfg.dropout, self.training)
            ### Begin CompleteP code ###
            # CompleteP scales the attention branch by depth; NanoGPT uses 1.
            x = x + cfg.attn_residual_multiplier * a
            ### End CompleteP code ###

            # 4. Pre-Norm → linear → GELU → linear → residual addition.
            ### Begin normalization-ablation code ###
            u = self._norm(x, f'h{i}_mlp_norm')
            ### End normalization-ablation code ###
            f = F.linear(u, p[f'h{i}_mlp_in_weight'], p[f'h{i}_mlp_in_bias'])
            f = F.gelu(f)
            ### Begin normalization-ablation code ###
            # W_2 input: normalize the 4N features after GELU.
            if cfg.extra_output_norm:
                f = self._norm(f, f'h{i}_mlp_out_norm')
            ### End normalization-ablation code ###
            f = F.linear(f, p[f'h{i}_mlp_out_weight'], p[f'h{i}_mlp_out_bias'])
            f = F.dropout(f, cfg.dropout, self.training)
            ### Begin CompleteP code ###
            # CompleteP scales the MLP branch by depth; NanoGPT uses 1.
            x = x + cfg.mlp_residual_multiplier * f
            ### End CompleteP code ###

        # 5. Final norm and the shared vocabulary projection.
        ### Begin normalization-ablation code ###
        x = self._norm(x, 'final_norm')
        ### End normalization-ablation code ###
        if targets is None:
            return F.linear(x[:, [-1], :], p['head_weight']), None
        ### Begin CompleteP code ###
        # Match the CompleteP author's output scaling only when targets are supplied.
        # This also applies during loss evaluation; NanoGPT uses 1.
        x = x * cfg.training_output_multiplier
        ### End CompleteP code ###
        logits = F.linear(x, p['head_weight'])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             adam_eps=1e-8, group_settings=None):
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
            ### Begin normalization-ablation code ###
            if self.config.extra_output_norm:
                for name in (f'h{i}_attn_out_norm', f'h{i}_mlp_out_norm'):
                    parameter_roles.extend([
                        (p[name + '_weight'], 'hidden_norm'),
                        (p[name + '_bias'],   'hidden_norm'),
                    ])
            ### End normalization-ablation code ###
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
        ### Begin CompleteP code ###
        # CompleteP updates exactly the same parameters as NanoGPT; none are added.
        # Keep five roles separate to assign per-role lr_scale, eps and weight_decay.
        # Hidden Norm/bias and final Norm can need different update settings.
        if group_settings is not None:
            if set(group_settings) != set(group_for_role):
                raise ValueError(f'group_settings must specify these roles: {tuple(group_for_role)}')
            group_for_role = {role: role for role in group_for_role}
            update_settings = group_settings
        # AdamW uses each group's eps/weight_decay directly. lr_scale is metadata:
        # train.py sets group['lr'] = scheduled_lr * lr_scale before each step.
        ### End CompleteP code ###

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
