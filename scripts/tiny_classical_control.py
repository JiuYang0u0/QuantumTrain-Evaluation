"""Train-only capacity control: same twenty examples, optimizer, and epochs."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from qt_eval.data import load_digits_splits
from qt_eval.models import ClassicalMLP
from qt_eval.artifacts import write_json


def run(output):
    x,y=load_digits_splits()['train'].tensors
    idx=torch.cat([torch.where(y==i)[0][:2] for i in range(10)])
    x,y=x[idx],y[idx]
    rows=[]
    torch.set_num_threads(1)
    for seed in [11,22,33]:
        for kind in ['relu','leaky_relu']:
            torch.manual_seed(seed)
            model=ClassicalMLP(kind).double()
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            history=[]
            for epoch in range(200):
                opt.zero_grad()
                loss=F.cross_entropy(model(x),y)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    history.append(float((model(x).argmax(1)==y).double().mean()))
            rows.append(dict(seed=seed,target=kind,final=history[-1],maximum=max(history)))
    path=Path(output)
    if path.exists():
        raise FileExistsError('Choose a new output path to preserve existing evidence')
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,dict(epochs=200,lr=.001,samples=20,scope='train-only follow-up control',results=rows))
    print(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/classical-tiny-control.json')
    run(parser.parse_args().output)
