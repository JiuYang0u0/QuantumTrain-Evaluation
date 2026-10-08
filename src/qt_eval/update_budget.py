"""Exactly matched updates and examples seen, via sampling with replacement."""
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .experiment import evaluate
from .artifacts import write_json


def train_updates(model,splits,output,updates=4000,batch_size=128,eval_every=100,seed=11,milestones=()):
    if set(splits)!={'train','validation'}:raise ValueError('Only train/validation accepted')
    if min(updates,batch_size,eval_every)<1:raise ValueError('Positive budgets required')
    if any(type(m)is not int or m<1 or m>updates or (m%eval_every and m!=updates) for m in milestones):
        raise ValueError('Milestones must be scheduled evaluation updates')
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    x,y=splits['train'].tensors
    generator=torch.Generator().manual_seed(seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    seen=torch.zeros(len(x),dtype=torch.bool)
    history=[];best=float('inf');elapsed=0.
    for step in range(1,updates+1):
        idx=torch.randint(len(x),(batch_size,),generator=generator);seen[idx]=True
        model.train();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[idx]),y[idx])
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise RuntimeError('Invalid gradient')
        qgrad=float(model.theta.grad.norm()) if hasattr(model,'theta') else None
        optimizer.step();elapsed+=time.perf_counter()-start
        if step%eval_every==0 or step==updates:
            with torch.random.fork_rng(devices=[]):
                val=evaluate(model,DataLoader(splits['validation'],batch_size=batch_size))
            history.append(dict(update=step,last_batch_loss=float(loss.detach()),validation=val,quantum_last_batch_grad_norm=qgrad))
            write_json(path/'history.json',history)
            if val['loss']<best:
                best=val['loss'];torch.save(dict(model_state=model.state_dict(),update=step),path/'best.pt')
            if step in milestones:
                torch.save(dict(model_state=model.state_dict(),update=step,validation=val),path/f'update-{step}.pt')
                print('milestone',step,'validation',val,flush=True)
    loaders={k:DataLoader(v,batch_size=batch_size) for k,v in splits.items()}
    final={k:evaluate(model,v) for k,v in loaders.items()}
    checkpoint=torch.load(path/'best.pt',weights_only=True);model.load_state_dict(checkpoint['model_state'])
    return dict(updates=updates,examples_seen=updates*batch_size,unique_examples_seen=int(seen.sum()),
        history=history,best_update=checkpoint['update'],final=final,best={k:evaluate(model,v) for k,v in loaders.items()},train_seconds=elapsed)
