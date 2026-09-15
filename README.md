# MiniP

MiniP is a small learning and experimentation repository derived from NanoGPT. Its focus is making **GPT's forward computation, parameter initialization, automatic differentiation, and AdamW updates** easy to trace in a few Python files.

<!-- ### Begin normalization-ablation code ### -->
**This checkout: `normalization-ablation`.** Branched from `completep` at `f470927`, this version adds selectable LayerNorm/RMSNorm and optional normalization immediately before both W_O and W_2. Defaults retain the previous LayerNorm computation. CompleteP settings remain available; they are not enabled by the branch name.

### Four normalization configurations

| Configuration | `norm_type` | `extra_output_norm` |
| --- | --- | --- |
| LN | `'layernorm'` | `False` |
| LN + extra | `'layernorm'` | `True` |
| RMS | `'rmsnorm'` | `False` |
| RMS + extra | `'rmsnorm'` | `True` |

`norm_type` selects the two existing pre-norms in every block and the final norm. With `extra_output_norm=True`, each block also normalizes the concatenated attention output over N features before W_O, and the GELU output over 4N features before W_2. Each added norm has its own trainable scale, initialized via `norm_weight_init`. All norm parameters use the existing hidden-norm optimizer role, without weight decay under the neutral/default optimizer settings.

`norm_eps` defaults to `1e-5` for both types. RMSNorm has no beta; `bias` still controls Linear biases and LayerNorm beta. For the four-way comparison with matching baseline parameter counts, explicitly set `bias=False`.

Notebook callers can pass these fields to `GPTConfig`, or obtain a training configuration with `get_config([])`, set the fields, and call `train.main(cfg)`. The CLI also accepts the three fields. Saved model configurations include them so sampling reconstructs the same architecture.

Every branch-specific code change is enclosed by `### Begin normalization-ablation code ###` and `### End normalization-ablation code ###`. The new options do not change matrix initialization distributions or their random draw order.

Validation covered CPU and MPS: baseline parity against `f470927`, all four variants with Linear bias on/off, an independent forward expression, shared matrix initialization, optimizer membership and added-scale updates, and trainer/checkpoint configuration round-trips. These are small correctness checks, not training-performance results.

### Saved normalization experiments

Only these three experiment notebooks are included, with their saved outputs:

| Notebook | Experiment |
| --- | --- |
| [07](notebooks/07-shakespeare-normalization-ablation.ipynb) | Four normalization variants at 4 layers; all reached 5,000 updates |
| [08](notebooks/08-completep-depth-normalization-ablation.ipynb) | CompleteP depth sweep; all four variants at depths 2–64 reached 1,000 updates; the 128-layer stage was paused |
| [09](notebooks/09-residual-scaling-fixed-lr-ablation.ipynb) | Unscaled vs. scaled residuals at depths 2–64, constant LR 0.0003; 44 actual runs reached 1,000 updates, representing 48 conditions with depth-2 reuse |

Notebook 09 ends with an embedded six-row overview comparing LayerNorm/RMSNorm with and without extra normalization. Its final, self-contained plotting cell exports a PNG and a single-page A3 PDF from saved metrics. Notebook 08 retains the intentional interruption output from its controlled stop.

The notebooks preserve the local execution configuration and saved figures. Raw metrics, datasets, checkpoints, and exported image/PDF files remain excluded from Git. To rerun or resume, configure the data, device, and result paths for your machine; existing resume paths refer to the original local runs. Regenerating reports requires the corresponding metrics files. Supporting scripts cover notebook execution, the Notebook 08 continuation queue, and clearer per-depth Notebook 09 reports.

## Branches
<!-- ### End normalization-ablation code ### -->

| Branch | Purpose | What to compare |
| --- | --- | --- |
| `nanogpt-upstream` | Unmodified NanoGPT snapshot at `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | The original implementation and README |
| `main` | Simplified NanoGPT baseline with persistent parameter tensors and an explicit functional forward | How the original model and training loop were made easier to read |
| `completep` | Numerical parameterization settings for the author's NanoGPT-based CompleteP implementation | Initialization, forward scaling, and optimizer-group differences |
<!-- ### Begin normalization-ablation code ### -->

`normalization-ablation` extends `completep` with the four normalization configurations above. Compare its model/trainer changes against `completep`.
<!-- ### End normalization-ablation code ### -->

`main` and `completep` share a simplified code lineage and are maintained as parallel versions. `completep` is not necessarily a descendant of the latest `main` commit. The upstream branch stays fixed as a reference.

```bash
git switch main
git diff nanogpt-upstream main -- model.py train.py sample.py
git diff main completep -- model.py train.py
```

## Read the algorithm through the code

Start with **`GPT.forward()`**, then follow the loss into the training loop and optimizer update.

| Step | Where to read | What happens |
| --- | --- | --- |
| Create parameters | [GPT.__init__](model.py#L74) | Register persistent weight/bias tensors in `nn.ParameterDict`; tie the token embedding and output head |
| Initialize values | [initialize_parameters](model.py#L122) | Set matrix standard deviations, norm scales/offsets, and linear biases |
| Forward | [GPT.forward](model.py#L202) | Token/position embeddings, repeated attention and MLP residual updates, final norm, logits and loss |
| Configure updates | [configure_optimizers](model.py#L285) | List parameter tensors and roles, assign group settings, return a PyTorch AdamW object |
| Backward and update | [train.main](train.py#L101) | Accumulate loss gradients, optionally clip them, run AdamW, and clear gradients |
| Generate text | [sample.py](sample.py) and [GPT.generate](model.py#L388) | Load a saved checkpoint and generate tokens autoregressively |

### Forward: one visible chain

For token IDs `x` of shape `[B, T]`, the hidden state has shape `[B, T, N]`. The model uses learned absolute position embeddings, Pre-LayerNorm, GELU, and a shared input/output embedding weight.

```text
token IDs
  -> token embedding + learned position embedding -> dropout
  -> repeat for each layer:
       LayerNorm -> QKV projection -> causal attention -> output projection
       -> dropout -> residual addition
       LayerNorm -> MLP expansion -> GELU -> MLP output projection
       -> dropout -> residual addition
  -> final LayerNorm -> vocabulary projection
  -> logits [B, T, V] and next-token cross-entropy loss
```

Without targets, forward returns logits only for the last position, `[B, 1, V]`, and `None` for the loss. This supports the sampling loop.

`model.py` stores parameters separately from their computation. `GPT.forward()` calls `torch.nn.functional` operations directly; it does not construct new trainable layers on each call. Temporary `nn.Linear`/`nn.Embedding` constructors are used during model construction to create tensors, not to hide the forward chain.

### Backward and AdamW: the same tensors throughout

```mermaid
flowchart TD
    D[Token IDs and targets] --> F[GPT.forward]
    P[Persistent model parameters] --> F
    F --> L[Cross-entropy loss]
    L --> B[Autograd backward]
    B --> G[Each parameter's .grad]
    P --> C[configure_optimizers: parameter references and settings]
    C --> O[PyTorch AdamW]
    G --> U[Optimizer step]
    O --> U
    U --> P
```

The diagram separates optimizer construction, done once, from updates, performed repeatedly. Conceptually a training iteration is:

```python
logits, loss = model(x, targets)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

The actual trainer also divides the loss across gradient-accumulation microbatches, optionally clips gradients, and uses `GradScaler` calls around backward/step. With the documented CPU/MPS `float32` runs, gradient scaling is disabled.

There is no handwritten backward method or custom AdamW implementation. PyTorch autograd writes gradients into each parameter's `.grad`. AdamW holds **references to those same parameters**, tracks their moment estimates, and updates them in place when a step occurs. The next forward reads the updated values.

### Parameter roles and optimizer groups

The optimizer function lists each parameter once, alongside its role. It does not infer roles using name prefixes or suffixes. Optional biases that are `None` and parameters with `requires_grad=False` are omitted; the shared head is counted once.

| Parameter role | `main`: NanoGPT grouping | `completep`: when role settings are supplied |
| --- | --- | --- |
| Token/position embedding, including the shared head | Weight decay | Embedding group |
| Hidden QKV, attention output, MLP input/output matrices | Weight decay | Hidden-weight group |
| Hidden LayerNorm gamma/beta | No weight decay | Hidden-norm group |
| Hidden linear biases | No weight decay | Hidden-bias group |
| Final LayerNorm gamma/beta | No weight decay | Final-norm group |

**CompleteP does not add a new set of trainable parameters through optimizer grouping.** It assigns different update settings to the existing parameters.

`lr`, `betas`, and `eps` passed to the AdamW constructor are defaults. A parameter group can override them. In the CompleteP branch, `lr_scale` is an additional field interpreted by **the training loop**, not automatically by AdamW:

```python
group['lr'] = scheduled_lr * group.get('lr_scale', 1.0)
```

Without CompleteP group settings, the experiment branch also uses the NanoGPT two-group scheme. The baseline branch has no `group_settings` argument.

## What was simplified from NanoGPT

| Area | Changes in the simplified branches |
| --- | --- |
| Model readability | Replaced nested custom Block/Attention/MLP computation with an explicit functional forward over registered parameter tensors |
| Initialization | Exposed matrix standard deviations and norm/bias initial values through `GPTConfig`; put initialization in a dedicated method |
| Optimizer readability | Made the parameter roster and update-group assignment explicit |
| Training entry point | Kept settings in `get_config()` and a single-device scratch-training workflow in `main()` |
| Removed training paths | No resume training, pretrained GPT-2 initialization, DDP/multi-process training, `torch.compile` toggle, or W&B logging in `train.py` |
| Removed model utilities | No pretrained-weight loader, context-cropping method, or model MFU estimator |
| Removed analysis notebooks | Removed upstream `scaling_laws.ipynb` and `transformer_sizing.ipynb` from the working branches |

Data preparation, gradient accumulation, optional gradient clipping, validation loss, warmup/cosine scheduling, checkpoint saving, and sampling remain. **No resume training does not mean no checkpoint loading:** `sample.py` still loads saved weights for generation, including supported earlier checkpoint layouts.

`bench.py`, `configurator.py`, some upstream presets, dataset helpers, and assets remain. The old pretrained/finetuning presets are not supported entry points for this scratch trainer. Use the explicit commands below rather than assuming every inherited preset is compatible. `bench.py` is an auxiliary benchmark, not part of the train-to-checkpoint-to-sample path.

## CompleteP-specific additions

The `completep` branch adds caller-supplied values for:

- QKV and MLP expansion initialization standard deviations, alongside configurable output-projection standard deviations.
- Attention score scaling (`1/D` for the author's muP setup, versus NanoGPT's default `1/sqrt(D)`), where `D` is the head width.
- Input, attention-residual, MLP-residual, and training-output multipliers.
- Per-role learning-rate scales, Adam epsilon, and weight decay.

Width/depth formulas are computed by the caller or experiment Notebook and passed as numbers. The model does not derive the full paper parameterization automatically from width and depth. Relevant additions are enclosed by `### Begin CompleteP code ###` and `### End CompleteP code ###` comments.

The author's implementation applies output scaling when targets are supplied, including loss evaluation, but omits it on the no-target sampling path. MiniP's experiment branch follows that behavior.

There is also an initialization-order difference between the working branches: `main` retains the baseline's repeated initialization order, whereas `completep` initializes each shared tensor once in a layer-local pass. Even with neutral forward scales, their independent same-seed initial weights need not match. This is distinct from matching computations and updates **after loading the same weights**.

## Devices and execution scope

The documented small runs have been exercised on CPU and Apple Silicon MPS using `float32`. Development was on an M4 Max with Python 3.13.7 and PyTorch 2.14.0.

CUDA code remains, including autocast, optional gradient scaling and CUDA-only fused AdamW selection. CUDA was not validated in the current local checks. The trainer's inherited default is still `device='cuda'`, so **explicitly select `mps` or `cpu` and `float32` on a Mac**. Multi-GPU/DDP training has been removed.

## Quick start: a small scratch run

These commands apply to `main` and to the neutral/default settings on `completep`. They are a pipeline check, not a CompleteP paper reproduction.

In a fresh checkout, create an environment and install the direct dependencies for this example:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch==2.14.0 numpy==2.5.3 requests==2.34.2 tiktoken==0.14.0
.venv/bin/python data/shakespeare_char/prepare.py
```

Train on MPS:

```bash
.venv/bin/python train.py \
    --dataset=shakespeare_char \
    --out_dir=out-readme-smoke \
    --device=mps \
    --dtype=float32 \
    --n_layer=2 \
    --n_head=2 \
    --n_embd=128 \
    --block_size=128 \
    --batch_size=8 \
    --gradient_accumulation_steps=1 \
    --dropout=0.0 \
    --learning_rate=0.001 \
    --decay_lr=False \
    --max_iters=100 \
    --eval_interval=50 \
    --eval_iters=10 \
    --log_interval=10
```

Generate from the checkpoint:

```bash
.venv/bin/python sample.py \
    --out_dir=out-readme-smoke \
    --device=mps \
    --dtype=float32 \
    --num_samples=1 \
    --max_new_tokens=200 \
    --temperature=0.8 \
    --top_k=40
```

For CPU, replace `--device=mps` with `--device=cpu` in both commands. Do not pass removed options such as `--init_from`, `--compile`, or `--wandb_log`.

The model is about 0.40M parameters. Expect finite losses, a checkpoint, and generated character text; coherent language is not the goal of this short run. The loop retains NanoGPT's step convention: `max_iters=100` performs updates numbered 0 through 100, with evaluation/checkpoint saving before the corresponding update. The final saved checkpoint is therefore from before update 100.

## Verification and local artifacts

Verification results have different scopes:

| Check | What it establishes | Limit |
| --- | --- | --- |
| Baseline versus pinned NanoGPT | Initialization, forward, gradients and updates in the recorded settings | Earlier MPS tests with nonzero dropout also showed upstream self-repeatability differences; not a blanket bitwise guarantee |
| CompleteP shared-weight alignment | Matching computations and updates against the pinned author implementation | Does not establish identical independent same-seed initialization after the layer-local initialization change |
| Optimizer roster refactor | 20 experiment-branch and 6 baseline-branch CPU/MPS cases passed, including group order, frozen parameters, three update steps and optimizer-state loading | Checks each branch before/after the optimizer change; not a full training reproduction |
| Reduced depth experiment | A small Figure 7-style observation was run previously | Did not reproduce every paper phenomenon; no claim of full Figure 7 reproduction |

Notebooks, their outputs, local Chinese study notes, checkpoints, and generated datasets are not distributed with the working branches. In the author's local checkout, the latest optimizer check is `notebooks/04-optimizer-grouping-check.ipynb`; other runs record their own source versions. These paths are local references, not files expected in a fresh clone.

The `notebooks/`, `docs/`, and `results/` directories are Git-ignored. Local study artifacts were removed from the published working-branch history; the untouched upstream reference still contains its original public files. Historical experiment tags remain local unless explicitly shared.

## Sources and license

- [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT), pinned at `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` for the upstream branch.
- [The author's NanoGPT-based muP/CompleteP implementation](https://github.com/EleutherAI/nanoGPT-mup), inspected at `88458c3f063b5c3c573f87435dc91f680174f0da`.
- [CompleteP paper](https://arxiv.org/abs/2505.01618).

The code retains the upstream [MIT license](LICENSE). The upstream branch's original README documents the original project, not the reduced workflow on the two working branches.
