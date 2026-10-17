"""Run one reproduction command and preserve output, timing and exit status."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--log-dir', required=True)
p.add_argument('--name', required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command and a.command[0] == '--' else a.command
if not command:
    p.error('A command is required')
if command[0] == 'python':
    command[0] = sys.executable
d = Path(a.log_dir)
d.mkdir(parents=True, exist_ok=True)
log_path, receipt_path = d / (a.name + '.log'), d / (a.name + '.json')
if log_path.exists() or receipt_path.exists():
    p.error('Log exists; use a new name to preserve evidence')
started = dt.datetime.now(dt.timezone.utc).isoformat()
tick = time.perf_counter()
env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
with log_path.open('w', encoding='utf-8') as log:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', env=env)
    for line in proc.stdout:
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
    code = proc.wait()
receipt = dict(command=command, cwd=str(Path.cwd()), started_utc=started,
               finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               elapsed_seconds=round(time.perf_counter()-tick, 3), exit_code=code)
receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
sys.exit(code)
