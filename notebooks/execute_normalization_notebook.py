### Begin normalization-ablation code ###
# Branch: codex/normalization-ablation.
# Execute the actual notebook and persist outputs while training continues.
import json
import os
import sys
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(sys.argv[1]).resolve()
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
LOG_DIR = ROOT / 'results' / 'notebook-executions' / run_id
LOG_DIR.mkdir(parents=True, exist_ok=False)

# Use a temporary Jupyter kernel definition pointing to this exact virtualenv.
# Nothing is installed into the user's Jupyter configuration.
runtime = Path('/private/tmp') / ('minip-jupyter-' + run_id)
kernel_dir = runtime / 'kernels' / 'minip-normalization'
kernel_dir.mkdir(parents=True)
(kernel_dir / 'kernel.json').write_text(json.dumps(dict(
    argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    display_name='MiniP experiment virtualenv', language='python',
)))
os.environ['JUPYTER_PATH'] = str(runtime) + (os.pathsep + os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime / 'runtime')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
os.environ['XDG_CACHE_HOME'] = str(runtime / 'cache')

import nbformat
from nbclient import NotebookClient

nb = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(nb)
(LOG_DIR / (NOTEBOOK.stem + '-before-execution.ipynb')).write_text(nbformat.writes(nb))
status = dict(started_utc=datetime.now(timezone.utc).isoformat(), notebook=str(NOTEBOOK),
              python=sys.executable, runner_pid=os.getpid(), status='running', completed_cells=[])
log = (LOG_DIR / 'execution.log').open('a', buffering=1)

def save():
    temporary = NOTEBOOK.with_suffix('.executing.tmp.ipynb')
    temporary.write_text(nbformat.writes(nb))
    temporary.replace(NOTEBOOK)
    (LOG_DIR / 'status.json').write_text(json.dumps(status, indent=2))

class RecordingClient(NotebookClient):
    last_saved = 0.0

    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg['msg_type'] == 'stream':
            text = msg['content'].get('text', '')
            print(text, end='', flush=True)
            log.write(text)
        # Persist growing training output as well as completed-cell outputs.
        if time.monotonic() - self.last_saved > 10:
            save()
            self.last_saved = time.monotonic()
        return result

    async def async_execute_cell(self, cell, cell_index, *args, **kwargs):
        status['current_cell'] = cell_index
        provisioner = getattr(self.km, 'provisioner', None)
        status['kernel_pid'] = getattr(provisioner, 'pid', None)
        try:
            result = await super().async_execute_cell(cell, cell_index, *args, **kwargs)
        except BaseException:
            save()
            raise
        status['completed_cells'].append(cell_index)
        save()
        return result

def stop_requested(signum, frame):
    raise KeyboardInterrupt('Notebook execution stopped by signal')

signal.signal(signal.SIGTERM, stop_requested)
print('Execution log directory:', LOG_DIR, flush=True)
save()
try:
    client = RecordingClient(nb, timeout=None, kernel_name='minip-normalization',
                             resources={'metadata': {'path': str(ROOT)}},
                             allow_errors=False, record_timing=True)
    client.execute(cwd=str(ROOT))
except BaseException:
    status['status'] = 'failed'
    status['error'] = traceback.format_exc()
    print(status['error'], flush=True)
    raise
else:
    status['status'] = 'complete'
    (LOG_DIR / (NOTEBOOK.stem + '-executed.ipynb')).write_text(nbformat.writes(nb))
    print('NOTEBOOK EXECUTION COMPLETE:', NOTEBOOK, flush=True)
finally:
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    log.close()

### End normalization-ablation code ###
