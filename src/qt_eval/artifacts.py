"""Portable run metadata and atomic JSON writes."""
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

SOURCE_COMMIT = 'd5b654116565ae9550e540e021ab0d4160a15c86'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def environment(config):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(['git',*args],cwd=root,capture_output=True,text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(python=platform.python_version(), platform=platform.platform(),
                cpu=platform.processor(), device='cpu', threads=config['torch_threads'],
                dtype=config['dtype'], source_commit=SOURCE_COMMIT,
                project_commit=git('rev-parse','HEAD'), working_tree=git('status','--porcelain'),
                packages={p:importlib.metadata.version(p) for p in
                          ['torch','pennylane','numpy','scikit-learn','pandas']})
