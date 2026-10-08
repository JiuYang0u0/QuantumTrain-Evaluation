"""Trace r1 CNN amplification; upstream-inspired global centering intervention."""
import argparse
import uuid
from pathlib import Path
import torch
from torch.nn import functional as F
from qt_eval.paper_mnist import PaperQuantumTrain,TargetCNN,unpack,functional_cnn
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.artifacts import write_json


def stats(t):
    t=t.detach()
    return dict(mean=float(t.mean()),std=float(t.std(unbiased=False)),absmax=float(t.abs().max()))


def trace(x,w):
    weights=unpack(w);layers={}
    x=F.conv2d(x,weights['conv1.weight'],weights['conv1.bias']);layers['conv1']=stats(x)
    x=F.max_pool2d(x,2);layers['pool1']=stats(x)
    x=F.conv2d(x,weights['conv2.weight'],weights['conv2.bias']);layers['conv2']=stats(x)
    x=F.max_pool2d(x,2);layers['pool2']=stats(x)
    x=F.linear(x.flatten(1),weights['fc1.weight'],weights['fc1.bias']);layers['fc1']=stats(x)
    x=F.linear(x,weights['fc2.weight'],weights['fc2.bias']);layers['logits']=stats(x)
    return dict(weights={k:stats(v) for k,v in weights.items()},layers=layers,
                logit_class_spread=stats(x-x.mean(1,keepdim=True)))


def run(output,data_root):
    root=Path(output)/('paper-scale-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    data=load_mnist_pilot(data_root);x,y=data['train'].tensors;x,y=x[:128],y[:128]
    write_json(root/'protocol.json',dict(seeds=[11,22,33],probe='first128 of fixed1000 MNIST train subset',
        conditions=['direct_cnn','r1_raw','global_center'],test_access=False,
        intervention='subtract mean across all6690 generated weights AND biases, as upstream MNIST code',
        scope='initialization diagnostic only; not training accuracy'))
    rows=[];old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in [11,22,33]:
            torch.manual_seed(seed);direct=TargetCNN().double()
            torch.manual_seed(seed);qt=PaperQuantumTrain()
            for condition in ['direct_cnn','r1_raw','global_center']:
                qt.zero_grad(set_to_none=True)
                w=torch.cat([p.flatten() for p in direct.parameters()]) if condition=='direct_cnn' else qt.generated_weights()
                if condition=='global_center':w=w-w.mean()
                logits=functional_cnn(x,w);loss=F.cross_entropy(logits,y)
                observation=trace(x,w)
                if condition!='direct_cnn':
                    gradient=torch.autograd.grad(loss,[qt.theta,*qt.mapping.parameters()])
                    observation['quantum_grad_norm']=float(gradient[0].norm())
                    observation['all_gradients_finite']=all(bool(torch.isfinite(g).all()) for g in gradient)
                row=dict(seed=seed,condition=condition,loss=float(loss.detach()),observation=observation)
                rows.append(row);print(seed,condition,'loss',row['loss'],'fc1 absmax',observation['layers']['fc1']['absmax'],flush=True)
        write_json(root/'diagnosis.json',rows)
    finally:torch.set_num_threads(old)
    print(root,flush=True);return root


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results')
    p.add_argument('--data',default='data/mnist');a=p.parse_args();run(a.output,a.data)
