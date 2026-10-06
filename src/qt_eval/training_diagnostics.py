"""Validation-only mapping study with observational training traces."""
import argparse
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from .quantum import QuantumTrain
from .data import load_digits_splits
from .experiment import evaluate
from .config import validate_config
from .artifacts import write_json,environment


def snapshot(model,x):
    """Observe a fixed training probe without RNG consumption or model mutation."""
    with torch.no_grad():
        weights=model.generated_weights()
        hidden=F.relu(F.linear(x,weights[:512].reshape(8,64),weights[512:520]))
        logits=F.linear(hidden,weights[520:600].reshape(10,8),weights[600:])
        def stats(t):
            return dict(count=t.numel(),mean=float(t.mean()),std=float(t.std(unbiased=False)),
                        minimum=float(t.min()),maximum=float(t.max()))
        return dict(weights=stats(weights),logits=stats(logits),
                    logits_between_images_std=float(logits.std(dim=0,unbiased=False).mean()),
                    prediction_counts=torch.bincount(logits.argmax(1),minlength=10).tolist(),
                    relu_zero_fraction=float((hidden==0).double().mean()),
                    hidden_active_fraction=(hidden>0).double().mean(0).tolist())


def run_diagnostic(activation,seed,splits,output,epochs=20,tiny=False):
    """Only train/validation are accessed. Tiny subset has two examples per class."""
    config=validate_config(dict(mode='full',mapping_activation=activation,seed=seed,epochs=epochs,q_depth=3))
    path=Path(output)/(f'{activation}-seed{seed}-'+('tiny' if tiny else 'full'))
    path.mkdir(parents=True,exist_ok=False)
    write_json(path/'config.json',config|{'tiny':tiny,'purpose':'validation-only diagnostics'})
    write_json(path/'result.json',{'status':'running'})
    previous_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model=QuantumTrain(3,mapping_activation=activation)
            train=splits['train']
            x,y=train.tensors
            if tiny:
                idx=torch.cat([torch.where(y==label)[0][:2] for label in range(10)])
                if len(idx)!=20:
                    raise ValueError('Tiny experiment needs at least two training examples per class')
                x,y=x[idx],y[idx]
                train=TensorDataset(x,y)
            probe=x[:min(40,len(x))]
            loader=DataLoader(train,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
            validation=DataLoader(splits['validation'],batch_size=64)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            history=[]
            initial=snapshot(model,probe)
            best=float('inf')
            write_json(path/'environment.json',environment(config))
            for epoch in range(1,epochs+1):
                before_q=model.q_params.detach().clone()
                before_m=[p.detach().clone() for p in model.mapping.parameters()]
                qnorms,mnorms=[],[]
                elapsed=0.
                model.train()
                for bx,by in loader:
                    start=time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(bx),by)
                    if not torch.isfinite(loss):
                        raise RuntimeError('non-finite loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('missing/non-finite gradient')
                    qnorms.append(float(model.q_params.grad.norm()))
                    mnorms.append(sum(float(p.grad.norm())**2 for p in model.mapping.parameters())**.5)
                    optimizer.step()
                    elapsed+=time.perf_counter()-start
                train_metrics=evaluate(model,DataLoader(train,batch_size=64))
                val_metrics=evaluate(model,validation)
                observation=snapshot(model,probe)
                row=dict(epoch=epoch,train_loss=train_metrics['loss'],train_accuracy=train_metrics['accuracy'],
                         val_loss=val_metrics['loss'],val_accuracy=val_metrics['accuracy'],train_seconds=elapsed,
                         quantum_grad_norm=float(np.mean(qnorms)),mapping_grad_norm=float(np.mean(mnorms)),
                         quantum_max_delta=float((model.q_params.detach()-before_q).abs().max()),
                         mapping_max_delta=max(float((p.detach()-old).abs().max()) for p,old in zip(model.mapping.parameters(),before_m)),
                         observation=observation)
                history.append(row)
                write_json(path/'history.json',history)
                if val_metrics['loss'] < best:
                    best=val_metrics['loss']
                    torch.save({'model_state':model.state_dict(),'epoch':epoch,'val_loss':best},path/'best.pt')
            checkpoint=torch.load(path/'best.pt',weights_only=True)
            model.load_state_dict(checkpoint['model_state'])
            result=dict(status='completed',activation=activation,seed=seed,tiny=tiny,train_samples=len(train),
                        initial=initial,history=history,best_epoch=checkpoint['epoch'],
                        best_validation=evaluate(model,validation),best_train=evaluate(model,DataLoader(train,batch_size=64)),
                        final_observation=history[-1]['observation'],best_observation=snapshot(model,probe))
            write_json(path/'diagnostic.json',result)
            write_json(path/'result.json',{'status':'completed'})
            return result
    except BaseException as exc:
        import traceback
        (path/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(path/'result.json',{'status':'failed','error':str(exc)})
        raise
    finally:
        torch.set_num_threads(previous_threads)


def run_mapping_study(output,smoke=True):
    root=Path(output)/('mapping-'+uuid.uuid4().hex[:10])
    root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={k:data[k] for k in ('train','validation')}
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    write_json(root/'protocol.json',dict(smoke=smoke,split_seed=2026,depth=3,seeds=[11] if smoke else [11,22,33],
               tiny_epochs=2 if smoke else 200,full_epochs=2 if smoke else 20,test_access=False))
    rows=[]
    for tiny in (True,False):
        for seed in ([11] if smoke else [11,22,33]):
            for activation in ('affine','tanh'):
                print(f'Mapping {activation} seed={seed} tiny={tiny}',flush=True)
                result=run_diagnostic(activation,seed,splits,root,epochs=(2 if smoke else (200 if tiny else 20)),tiny=tiny)
                rows.append(dict(activation=activation,seed=seed,tiny=tiny,best_epoch=result['best_epoch'],
                                 train_accuracy=result['best_train']['accuracy'],val_accuracy=result['best_validation']['accuracy'],
                                 final_train_accuracy=result['history'][-1]['train_accuracy'],
                                 max_train_accuracy=max(h['train_accuracy'] for h in result['history']),
                                 train_seconds=sum(h['train_seconds'] for h in result['history'])))
                pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output',default='results')
    args=parser.parse_args()
    print(run_mapping_study(args.output,smoke=not args.full))
