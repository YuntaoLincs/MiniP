### Begin normalization-ablation code ###
# Branch: codex/normalization-ablation.
"""Render clearer reports from saved results without touching a running kernel."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/minip-notebook09-readable-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(run_dir, depth, target=1000):
    # Read-only scientific input. New presentation artifacts live in their own folder.
    run_dir = Path(run_dir).resolve()
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    report = metrics['depth_reports'][str(depth)]
    assert report['target_updates'] == target
    results = {r['id']: r for r in metrics['results']}
    entries = {(v['depth'], v['mode'], v['norm_id']): v for v in metrics['display_variants']}

    def history(mode, norm):
        result = results[entries[depth, mode, norm]['source_id']]
        assert result['completed_updates'] >= target
        rows = [r for r in result['evaluations'] if r['completed_updates'] <= target]
        assert rows[-1]['completed_updates'] == target
        return rows

    output = run_dir / 'figures' / f'{target}-updates' / 'readable-v2'
    output.mkdir(parents=True, exist_ok=True)
    names = {'A': 'LayerNorm', 'B': 'LayerNorm + extra',
             'C': 'RMSNorm', 'D': 'RMSNorm + extra'}
    styles = [dict(color='#0072B2', linestyle='-', marker='o', markevery=(0, 3)),
              dict(color='#D55E00', linestyle='--', marker='^', markevery=(1, 3))]

    def line(ax, rows, field, label, style):
        # Offset hollow markers keep near-identical curves legible without moving data.
        ax.plot([r['completed_updates'] for r in rows], [r[field] for r in rows],
                label=label, linewidth=2, markersize=6, markerfacecolor='white',
                markeredgewidth=1.5, **style)

    def format_axes(axes):
        for ax in axes.flat:
            ax.set(xlabel='Completed optimizer updates', ylabel='Loss (nats/token)')
            ax.grid(alpha=0.2)
            ax.legend(frameon=True, fontsize=10)

    def save(fig, stem):
        paths = {}
        for extension in ('png', 'pdf'):
            path = output / f'{stem}.{extension}'
            temporary = output / f'{stem}.tmp.{extension}'
            fig.savefig(temporary, dpi=170, format=extension)
            temporary.replace(path)
            paths[extension] = str(path)
        plt.close(fig)
        return paths

    figures = {}
    for mode in ('unscaled', 'scaled'):
        # Separate norm families, but share axes so visual comparisons remain fair.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True,
                                 constrained_layout=True)
        for row, (family, pair) in enumerate([('LayerNorm', 'AB'), ('RMSNorm', 'CD')]):
            for column, (field, split) in enumerate([('train_loss', 'Train (dropout off)'),
                                                    ('val_loss', 'Validation')]):
                ax = axes[row, column]
                for norm, label, style in zip(pair, ('Baseline', '+ extra norm'), styles):
                    line(ax, history(mode, norm), field, label, style)
                ax.set_title(f'{family} | {split}')
                ax.tick_params(labelbottom=True)
        format_axes(axes)
        residual = 1.0 if mode == 'unscaled' else 2.0 / depth
        fig.suptitle(f'Notebook 09 | {depth} layers | {mode}, residual={residual:g}\n'
                     f'Constant LR={metrics["settings"]["learning_rate"]:g} | '
                     'Norm families shown separately; shared axis scales', fontsize=14)
        figures[mode] = save(fig, f'L{depth:02d}-{mode}')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, norm in zip(axes.flat, 'ABCD'):
        for mode, style in zip(('unscaled', 'scaled'), styles):
            line(ax, history(mode, norm), 'val_loss', mode, style)
        ax.set_title(names[norm])
        ax.tick_params(labelbottom=True)
    format_axes(axes)
    fig.suptitle(f'Notebook 09 | {depth} layers | Residual scaling comparison\n'
                 'Validation loss | Solid circles: unscaled; dashed triangles: scaled', fontsize=14)
    figures['comparison'] = save(fig, f'L{depth:02d}-comparison')
    manifest = dict(depth=depth, target_updates=target, style='readable-v2', figures=figures,
                    png_paths=[figures[k]['png'] for k in ('unscaled', 'scaled', 'comparison')])
    path = output / f'L{depth:02d}-manifest.json'
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--target', type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(render(args.run_dir, args.depth, args.target), indent=2))
### End normalization-ablation code ###
