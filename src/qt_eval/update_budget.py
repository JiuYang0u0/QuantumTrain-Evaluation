"""Exactly matched updates and examples seen, via sampling with replacement."""
import time
import copy
import hashlib
import random
import numpy as np
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .experiment import evaluate
from .artifacts import write_json


def atomic_save(value,path):
    temporary=path.with_suffix('.pt.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def data_digest(splits):
    digest=hashlib.sha256()
    for key in ('train','validation'):
        for tensor in splits[key].tensors:
            digest.update(str((key,tuple(tensor.shape),tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_updates(model,splits,output,updates=4000,batch_size=128,eval_every=100,seed=11,milestones=(),resume=False,train_eval_size=None):
    if set(splits)!={'train','validation'}:raise ValueError('Only train/validation accepted')
    if min(updates,batch_size,eval_every)<1:raise ValueError('Positive budgets required')
    if any(type(m) is not int or m<1 or m>updates for m in milestones):
        raise ValueError('Milestones must be integers within the update budget')
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    protocol=dict(updates=updates,batch_size=batch_size,eval_every=eval_every,seed=seed,milestones=list(milestones),
        data_sha256=data_digest(splits),model_repr=repr(model),torch_version=str(torch.__version__),threads=torch.get_num_threads(),
        model_attributes={k:getattr(model,k,None) for k in ('global_center','probability_processing','target_activation','weight_transform','feature_scale')})
    if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('Resume protocol currently supports CPU only')
    if not resume and any((path/name).exists() for name in ('latest.pt','best.pt','history.json')):
        raise FileExistsError('Training artifacts exist; use resume=True or a new directory')
    x,y=splits['train'].tensors
    generator=torch.Generator().manual_seed(seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    seen=torch.zeros(len(x),dtype=torch.bool)
    history=[];best=float('inf');elapsed=0.;start_step=0;best_checkpoint=None
    if resume:
        checkpoint=torch.load(path/'latest.pt',weights_only=True)
        if checkpoint.get('schema')!=1 or checkpoint['protocol']!=protocol:
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(checkpoint['model_state']);optimizer.load_state_dict(checkpoint['optimizer_state'])
        generator.set_state(checkpoint['sampler_rng']);torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng'])
        ns=checkpoint['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        seen=checkpoint['seen'];history=checkpoint['history'];best=checkpoint['best_loss'];elapsed=checkpoint['train_seconds']
        start_step=checkpoint['step'];best_checkpoint=checkpoint['best_checkpoint']
        atomic_save(best_checkpoint,path/'best.pt');write_json(path/'history.json',history)
        print('Resuming from saved update',start_step,flush=True)
    for step in range(start_step+1,updates+1):
        idx=torch.randint(len(x),(batch_size,),generator=generator);seen[idx]=True
        model.train();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[idx]),y[idx])
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise RuntimeError('Invalid gradient')
        qgrad=float(model.theta.grad.norm()) if hasattr(model,'theta') else None
        optimizer.step();elapsed+=time.perf_counter()-start
        if step%eval_every==0 or step==updates or step in milestones:
            with torch.random.fork_rng(devices=[]):
                val=evaluate(model,DataLoader(splits['validation'],batch_size=batch_size))
            history.append(dict(update=step,last_batch_loss=float(loss.detach()),validation=val,quantum_last_batch_grad_norm=qgrad))
            write_json(path/'history.json',history)
            if val['loss']<best:
                best=val['loss'];best_checkpoint=dict(model_state=copy.deepcopy(model.state_dict()),update=step)
                atomic_save(best_checkpoint,path/'best.pt')
            if step in milestones:
                atomic_save(dict(model_state=model.state_dict(),update=step,validation=val),path/f'update-{step}.pt')
                print('milestone',step,'validation',val,flush=True)
            ns=np.random.get_state()
            atomic_save(dict(schema=1,protocol=protocol,step=step,model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),sampler_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
                python_rng=random.getstate(),numpy_rng=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                seen=seen,history=history,best_loss=best,best_checkpoint=best_checkpoint,train_seconds=elapsed),path/'latest.pt')
    def train_loader():
        if train_eval_size is None:
            return DataLoader(splits['train'],batch_size=batch_size)
        subset=torch.utils.data.TensorDataset(
            *[t[:train_eval_size] for t in splits['train'].tensors])
        return DataLoader(subset,batch_size=batch_size)
    final={'train':evaluate(model,train_loader()),
           'validation':evaluate(model,DataLoader(splits['validation'],batch_size=batch_size))}
    checkpoint=torch.load(path/'best.pt',weights_only=True);model.load_state_dict(checkpoint['model_state'])
    return dict(updates=updates,examples_seen=updates*batch_size,unique_examples_seen=int(seen.sum()),
        history=history,best_update=checkpoint['update'],final=final,
        best={'train':evaluate(model,train_loader()),
              'validation':evaluate(model,DataLoader(splits['validation'],batch_size=batch_size))},
        train_eval_size=train_eval_size,train_seconds=elapsed)
