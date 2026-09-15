"""Single-device training from scratch. Read get_config(), then main()."""

import math
import os
import pickle
import sys
import time
from ast import literal_eval
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model import GPT, GPTConfig


def get_config(argv=None):
    """Settings: defaults → optional Python preset → CLI overrides."""
    values = dict(
        # Model dimensions, context and dropout.
        n_layer=12, n_head=12, n_embd=768, block_size=1024,
        dropout=0.0, bias=False,
        ### Begin normalization-ablation code ###
        # Branch: codex/normalization-ablation. Shared by CLI and Notebook calls.
        norm_type='layernorm', extra_output_norm=False, norm_eps=1e-5,
        ### End normalization-ablation code ###
        # NanoGPT initialization: σ for embeddings, σ/√(2L) for output projections.
        # CompleteP presets can override each standard deviation.
        init_std=0.02, attn_out_init_std=None, mlp_out_init_std=None,
        ### Begin CompleteP code ###
        # CompleteP/μP width-scaled QKV and MLP initialization; None uses σ.
        qkv_init_std=None, mlp_in_init_std=None,
        ### End CompleteP code ###
        norm_weight_init=1.0, norm_bias_init=0.0, linear_bias_init=0.0,
        ### Begin CompleteP code ###
        # CompleteP forward scales; the preset/Notebook computes Table 1 formulas.
        attention_scale=None, input_multiplier=1.0,
        attn_residual_multiplier=1.0, mlp_residual_multiplier=1.0,
        training_output_multiplier=1.0,
        ### End CompleteP code ###
        # Base AdamW settings.
        learning_rate=6e-4, adam_eps=1e-8, weight_decay=0.1,
        beta1=0.9, beta2=0.95, grad_clip=1.0,
        ### Begin CompleteP code ###
        # CompleteP: role -> {lr_scale, eps, weight_decay}; None uses NanoGPT groups.
        group_settings=None,
        ### End CompleteP code ###
        # Training length and learning-rate schedule.
        max_iters=600000, decay_lr=True, warmup_iters=2000,
        lr_decay_iters=600000, min_lr=6e-5,
        # Dataset and batch size.
        dataset='openwebtext', batch_size=12, gradient_accumulation_steps=40,
        # Random seed, device and precision.
        seed=1337, device='cuda',
        dtype=('bfloat16' if torch.cuda.is_available()
               and torch.cuda.is_bf16_supported() else 'float16'),
        # Output and evaluation.
        out_dir='out', eval_interval=2000, eval_iters=200, log_interval=1,
        eval_only=False, always_save_checkpoint=True,
    )
    keys = set(values)
    for arg in sys.argv[1:] if argv is None else argv:
        if not arg.startswith('--'):
            overrides = dict(values)
            source = Path(arg).read_text()
            print(f'Overriding config with {arg}:\n{source}')
            exec(compile(source, arg, 'exec'), overrides)
            if 'init_from' in overrides:
                raise ValueError('This trainer always starts from scratch; remove init_from')
            values.update({k: overrides[k] for k in keys})
        else:
            key, text = arg[2:].split('=', 1)
            if key not in keys:
                raise ValueError(f'Unknown config key: {key}')
            try:
                value = literal_eval(text)
            except (SyntaxError, ValueError):
                value = text
            ### Begin CompleteP code ###
            # Extend optional numeric overrides to QKV/MLP stds and attention scale.
            if key in ('attn_out_init_std', 'mlp_out_init_std', 'qkv_init_std',
                       'mlp_in_init_std', 'attention_scale'):
                if value is not None and type(value) not in (int, float):
                    raise TypeError(f'{key} expects a number or None')
                value = None if value is None else float(value)
            elif key == 'group_settings':
                # CompleteP group settings are a dictionary.
                if value is not None and not isinstance(value, dict):
                    raise TypeError('group_settings expects a dictionary or None')
            ### End CompleteP code ###
            elif type(value) is not type(values[key]):
                raise TypeError(f'{key} expects {type(values[key]).__name__}')
            print(f'Overriding: {key} = {value}')
            values[key] = value
    return SimpleNamespace(**values)


def main(cfg=None):
    """Fresh model → optimizer → explicit forward/backward/update loop.

    Initialization settings enter GPTConfig; optimizer settings enter AdamW.
    Helpers below handle data, evaluation and disk output in this same file.
    """
    cfg = get_config() if cfg is None else cfg
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    precision = getattr(torch, cfg.dtype)
    ctx = (nullcontext() if device_type == 'cpu' else
           torch.amp.autocast(device_type=device_type, dtype=precision))
    tokens = cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps
    print(f'tokens per iteration will be: {tokens:,}')

    # Pass model dimensions, initialization and CompleteP forward scales to GPT.
    model_config = GPTConfig(
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
        block_size=cfg.block_size, vocab_size=get_vocab_size(cfg),
        dropout=cfg.dropout, bias=cfg.bias,
        ### Begin normalization-ablation code ###
        norm_type=cfg.norm_type, extra_output_norm=cfg.extra_output_norm,
        norm_eps=cfg.norm_eps,
        ### End normalization-ablation code ###
        init_std=cfg.init_std,
        attn_out_init_std=cfg.attn_out_init_std,
        mlp_out_init_std=cfg.mlp_out_init_std,
        ### Begin CompleteP code ###
        # Pass the width-scaled hidden initialization from the preset/Notebook.
        qkv_init_std=cfg.qkv_init_std, mlp_in_init_std=cfg.mlp_in_init_std,
        ### End CompleteP code ###
        norm_weight_init=cfg.norm_weight_init, norm_bias_init=cfg.norm_bias_init,
        linear_bias_init=cfg.linear_bias_init,
        ### Begin CompleteP code ###
        # Pass attention, input, residual and output scales to the forward chain.
        attention_scale=cfg.attention_scale, input_multiplier=cfg.input_multiplier,
        attn_residual_multiplier=cfg.attn_residual_multiplier,
        mlp_residual_multiplier=cfg.mlp_residual_multiplier,
        training_output_multiplier=cfg.training_output_multiplier,
        ### End CompleteP code ###
    )
    print('Initializing a new model from scratch')
    model = GPT(model_config).to(cfg.device)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == 'float16'))

    # Build AdamW with the base settings and optional CompleteP parameter groups.
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2),
        device_type, adam_eps=cfg.adam_eps,
        ### Begin CompleteP code ###
        # Supply per-role lr_scale, eps and weight_decay; None uses NanoGPT groups.
        group_settings=cfg.group_settings,
        ### End CompleteP code ###
    )

    X, Y = get_batch('train', cfg)
    best_val_loss = 1e9
    t0 = time.time()
    # Preserve upstream: max_iters=100 executes updates numbered 0..100.
    for step in range(cfg.max_iters + 1):
        lr = get_lr(step, cfg) if cfg.decay_lr else cfg.learning_rate
        ### Begin CompleteP code ###
        # CompleteP scales each group's scheduled learning rate; NanoGPT uses 1.
        for group in optimizer.param_groups:
            group['lr'] = lr * group.get('lr_scale', 1.0)
        ### End CompleteP code ###

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, cfg, ctx)
            print(f"step {step}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                if step > 0:
                    # Save pre-update weights for sampling and experiment inspection.
                    save_checkpoint(model, optimizer, cfg, step, best_val_loss)
        if step == 0 and cfg.eval_only:
            break

        # Forward → averaged microbatch loss → backward.
        for _ in range(cfg.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps
            # Keep prefetch before backward: this preserves the upstream RNG order.
            X, Y = get_batch('train', cfg)
            scaler.scale(loss).backward()

        # Clip accumulated gradients, update parameters, then clear gradients.
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt, t0 = t1 - t0, t1
        if step % cfg.log_interval == 0:
            lossf = loss.item() * cfg.gradient_accumulation_steps
            print(f'iter {step}: loss {lossf:.4f}, time {dt*1000:.2f}ms')


def get_vocab_size(cfg):
    """Read dataset vocabulary metadata; fall back to NanoGPT's padded GPT-2 size."""
    path = Path('data') / cfg.dataset / 'meta.pkl'
    if not path.exists():
        return 50304
    with path.open('rb') as file:
        size = pickle.load(file)['vocab_size']
    print(f'found vocab_size = {size} (inside {path})')
    return size


def get_batch(split, cfg):
    """Sample token windows and next-token targets; this consumes the CPU RNG."""
    path = Path('data') / cfg.dataset / ('train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+cfg.block_size].astype(np.int64)) for i in ix])
    if 'cuda' in cfg.device:
        return (x.pin_memory().to(cfg.device, non_blocking=True),
                y.pin_memory().to(cfg.device, non_blocking=True))
    return x.to(cfg.device), y.to(cfg.device)


@torch.no_grad()
def estimate_loss(model, cfg, ctx):
    """Average sampled train/validation losses, restoring training mode afterwards."""
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split, cfg)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(step, cfg):
    """Compute the baseline linear warmup followed by cosine decay."""
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / (cfg.warmup_iters + 1)
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (step - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= ratio <= 1
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.learning_rate - cfg.min_lr)


def save_checkpoint(model, optimizer, cfg, step, best_val_loss):
    """Persist weights/settings for sampling and optimizer state for inspection."""
    checkpoint = dict(
        model=model.state_dict(), optimizer=optimizer.state_dict(),
        model_args=asdict(model.config), iter_num=step,
        best_val_loss=best_val_loss, config=vars(cfg).copy(),
    )
    print(f'saving checkpoint to {cfg.out_dir}')
    torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt.pt'))


if __name__ == '__main__':
    main()
