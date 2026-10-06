"""Validate teaching structure and execute fresh kernels without rewriting sources."""
import argparse
import os
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=['本節目標','為什麼需要這一步','輸入／輸出與 shape','關鍵原理與程式定位',
          '操作步驟','預期觀察','結果判讀與常見錯誤','理解問題','<details>']


def verify(path,output):
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    sections=[c.source for c in notebook.cells if c.cell_type=='markdown' and c.source.startswith('## ')]
    assert sections, 'Missing teaching sections'
    for section in sections:
        assert all(term in section for term in REQUIRED), f'Incomplete teaching section: {section.splitlines()[0]}'
    setup=f"import os\nos.environ['QT_EVAL_MODE']='smoke'\nos.environ['QT_EVAL_OUTPUT']={str(output)!r}"
    notebook.cells.insert(0,nbformat.v4.new_code_cell(setup))
    NotebookClient(notebook,timeout=300,kernel_name='python3',
                   resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
    print(f'PASS {path.name}: {len(sections)} guided sections, fresh-kernel smoke',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path)
    parser.add_argument('--mode',choices=['smoke'],default='smoke')
    args=parser.parse_args()
    # Local temporary directory; no outputs or execution counts written to source notebooks.
    base=Path(os.environ.get('LOCALAPPDATA',tempfile.gettempdir()))/'Temp'/'opencode'
    base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qt-notebooks-',dir=base) as temp:
        paths=[args.notebook] if args.notebook else sorted((ROOT/'notebooks').glob('*.ipynb'))
        assert paths, 'No notebooks found'
        for path in paths:
            verify(path,Path(temp)/path.stem)
