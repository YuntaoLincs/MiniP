### Begin normalization-ablation code ###
# Branch: codex/normalization-ablation.
"""Execute Notebook 08 at successive total-update milestones, with safe resume."""
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/08-completep-depth-normalization-ablation.ipynb'
RESULTS = ROOT / 'results/completep-depth-normalization/20260914T200740Z-176b0e19'
STOP = RESULTS / 'STOP_REQUESTED'
TARGETS = (1000, 2000, 3000, 4000, 5000)
EXPECTED_IDS = {f'L{d:02d}-{v}' for d in (2, 4, 8, 16, 32, 64, 128) for v in 'ABCD'}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def configure_notebook(target):
    """Change only the total target, preserving the scientific training protocol."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    matches = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        cell.source, count = re.subn(r'\bmax_updates=\d+', f'max_updates={target}', cell.source)
        matches += count
        cell.outputs = []
        cell.execution_count = None
        cell.metadata.pop('execution', None)
    assert matches == 1, 'Expected exactly one editable max_updates setting.'
    nbformat.validate(notebook)
    temporary = NOTEBOOK.with_suffix('.stage.tmp.ipynb')
    nbformat.write(notebook, temporary)
    temporary.replace(NOTEBOOK)


def verify_and_archive(target):
    """Check actual optimizer counters, then preserve the milestone's reports."""
    metrics = json.loads((RESULTS / 'metrics.json').read_text())
    assert metrics['status'] == 'complete'
    assert metrics['settings']['max_updates'] == target
    assert {r['id'] for r in metrics['results']} == EXPECTED_IDS
    assert len(metrics['results']) == len(EXPECTED_IDS)
    checkpoints = []
    for result in metrics['results']:
        assert result['status'] == 'complete' and result['completed_updates'] == target
        assert [r['completed_updates'] for r in result['training']] == list(range(1, target + 1))
        assert [r['completed_updates'] for r in result['evaluations']] == list(range(0, target + 1, 50))
        path = RESULTS / f"{result['id']}-last.pt"
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['completed_updates'] == target
        assert saved['model_args']['n_layer'] == result['depth']
        assert saved['result']['training'] == result['training']
        assert saved['result']['evaluations'] == result['evaluations']
        assert {'python', 'numpy', 'torch_cpu', 'torch_mps'} <= saved['rng'].keys()
        assert {int(s['step'].item()) for s in saved['optimizer']['state'].values()} == {target}
        # Earlier checkpoints can retain the original five-depth signature at 1,000.
        old = saved['resume_signature']
        current = metrics['resume_signature']
        assert {k: v for k, v in old.items() if k != 'variants'} == {
            k: v for k, v in current.items() if k != 'variants'}
        assert old['variants'] == current['variants'][:len(old['variants'])]
        assert result['id'] in {v['id'] for v in old['variants']}
        checkpoints.append(dict(id=result['id'], completed_updates=target,
                                bytes=path.stat().st_size, val_loss=result['evaluations'][-1]['val_loss']))
        del saved

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == 'code' and c.source.strip()]
    assert all(c.execution_count is not None for c in code_cells)
    assert not [o for c in code_cells for o in c.outputs if o.output_type == 'error']
    figures = list(RESULTS.glob('*.png')) + list(RESULTS.glob('*.pdf'))
    assert len(figures) == 6
    destination = RESULTS / 'milestones' / f'{target}-updates'
    destination.mkdir(parents=True, exist_ok=True)
    for source in [RESULTS / 'metrics.json', NOTEBOOK, *figures]:
        temporary = destination / (source.name + '.tmp')
        shutil.copy2(source, temporary)
        temporary.replace(destination / source.name)
    # This marker is written last: an interrupted archive is never considered complete.
    atomic_json(destination / 'verified.json', dict(
        verified_utc=now(), target=target, checkpoints=checkpoints,
        weight_retention='Latest resumable checkpoint per run; historical metrics and figures per milestone.',
    ))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop', action='store_true', help='Request a safe stop between optimizer updates.')
    parser.add_argument('--resume', action='store_true', help='Clear a previous stop request and resume the unfinished stage.')
    args = parser.parse_args()
    if args.stop:
        STOP.write_text(f'Controlled stop requested at {now()}\n')
        print('Stop requested; the notebook will save at the next safe update boundary.', flush=True)
        return

    # Keep the lock for this process's lifetime; avoid two queues writing one experiment.
    lock = (RESULTS / '.stage-queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        STOP.unlink(missing_ok=True)
    if STOP.exists():
        raise SystemExit('A stop request is set. Use --resume when ready to continue.')

    queue_dir = ROOT / 'results/continuation-queues' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-depth-extension')
    queue_dir.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), started_utc=now(), status='running',
                 notebook=str(NOTEBOOK), results=str(RESULTS), targets=list(TARGETS), stages=[])

    def persist():
        atomic_json(queue_dir / 'status.json', state)
        atomic_json(RESULTS / 'active-stage-queue.json', dict(status_file=str(queue_dir / 'status.json'), **state))

    def request_stop(signum, frame):
        STOP.write_text(f'Signal {signum} requested a controlled stop at {now()}\n')
        state['status'] = 'stopping'
        persist()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    persist()
    print('STAGE QUEUE:', queue_dir, 'PID:', os.getpid(), flush=True)
    try:
        current_target = json.loads((RESULTS / 'metrics.json').read_text())['settings']['max_updates']
        for target in TARGETS:
            if target < current_target:
                continue
            if STOP.exists():
                state['status'] = 'stopped'
                break
            stage = dict(target=target, started_utc=now(), status='running')
            state['stages'].append(stage)
            state['active_target'] = target
            persist()
            configure_notebook(target)
            print(f'START: all 28 runs to {target} total updates.', flush=True)
            child = subprocess.Popen([
                sys.executable, '-u', str(ROOT / 'notebooks/execute_normalization_notebook.py'), str(NOTEBOOK)
            ], cwd=ROOT)
            state['active_runner_pid'] = child.pid
            persist()
            return_code = child.wait()
            state['active_runner_pid'] = None
            if STOP.exists():
                stage['status'] = state['status'] = 'stopped'
                break
            if return_code:
                raise RuntimeError(f'Notebook failed at milestone {target}: exit {return_code}. Queue halted.')
            archive = verify_and_archive(target)
            stage.update(status='complete', finished_utc=now(), archive=str(archive))
            persist()
            print(f'VERIFIED AND ARCHIVED: all 28 runs at {target} updates.', flush=True)
        else:
            state['status'] = 'complete'
    except BaseException:
        state['status'] = 'stopped' if STOP.exists() else 'failed'
        state['error'] = traceback.format_exc()
        raise
    finally:
        state['finished_utc'] = now()
        persist()
        print('STAGE QUEUE:', state['status'].upper(), flush=True)


if __name__ == '__main__':
    main()
### End normalization-ablation code ###
