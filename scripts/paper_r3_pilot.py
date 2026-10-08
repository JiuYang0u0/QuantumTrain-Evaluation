"""r2/r3 paired MNIST pilot: only probability postprocessing changes."""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from qt_eval.paper_mnist import PaperQuantumTrain
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.config import validate_config
from qt_eval.artifacts import write_json,environment
from qt_eval.experiment import fit,evaluate


def run(output='results',data_root='data/mnist',smoke=False):
    root=Path(output)/('paper-r3-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    protocol=dict(evidence='r2/r3 paired small-data pilot, not exact paper reproduction',smoke=smoke,
        seeds=[11] if smoke else [11,22,33],conditions=['raw','upstream'],epochs=2 if smoke else 50,
        train_samples=1000,validation_samples=200,batch_size=128,lr=.0001,blocks=16,
        parameters=457,target_parameters=6690,global_center=True,test_access=False,status='running',
        formula='f=(0.8*tanh(0.1*4096*p[:6690]))**0.3; f-=mean(f); no extra squaring or epsilon')
    write_json(root/'protocol.json',protocol)
    data=load_mnist_pilot(data_root);splits={k:data[k] for k in ('train','validation')}
    write_json(root/'split_indices.json',data['indices']);write_json(root/'dataset_hashes.json',data['hashes'])
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            for condition in protocol['conditions']:
                folder=root/f'{condition}-{seed}';folder.mkdir();print(condition,seed,flush=True)
                write_json(folder/'result.json',{'status':'running'})
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model=PaperQuantumTrain(global_center=True,probability_processing=condition)
                    config=validate_config(dict(seed=seed,epochs=protocol['epochs'],batch_size=128,learning_rate=.0001))
                    write_json(folder/'config.json',protocol|{'seed':seed,'condition':condition})
                    write_json(folder/'environment.json',environment(config))
                    x,y=splits['train'].tensors;x,y=x[:128],y[:128]
                    logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y)
                    grads=torch.autograd.grad(loss,[model.theta,*model.mapping.parameters()])
                    with torch.no_grad():
                        p=model.qnode(model.theta);f=model.mapping_inputs(p)[:,-1];w=model.generated_weights()
                        initial=dict(min_probability=float(p.min()),feature_mean=float(f.mean()),feature_std=float(f.std(unbiased=False)),
                            weight_std=float(w.std(unbiased=False)),logits_absmax=float(logits.abs().max()),loss=float(loss),
                            quantum_grad_norm=float(grads[0].norm()),mapping_grad_norm=sum(float(g.norm())**2 for g in grads[1:])**.5,
                            all_gradients_finite=all(bool(torch.isfinite(g).all()) for g in grads))
                    write_json(folder/'initial.json',initial)
                    if not initial['all_gradients_finite']:raise RuntimeError('Initial gradient non-finite; no automatic epsilon correction')
                    previous={k:p.detach().clone() for k,p in model.named_parameters()};updates=[]
                    def observe(m,epoch):
                        groups={}
                        for name,p in m.named_parameters():
                            group='quantum' if name=='theta' else 'mapping'
                            g=groups.setdefault(group,{'last_batch_grad_norm_squared':0.,'max_delta':0.})
                            g['last_batch_grad_norm_squared']+=float(p.grad.norm())**2
                            g['max_delta']=max(g['max_delta'],float((p.detach()-previous[name]).abs().max()))
                            previous[name]=p.detach().clone()
                        for g in groups.values():g['last_batch_grad_norm']=g.pop('last_batch_grad_norm_squared')**.5
                        updates.append(dict(epoch=epoch,groups=groups));write_json(folder/'updates.json',updates)
                    h=fit(model,splits,config,folder,epoch_observer=observe)
                    loaders={k:DataLoader(v,batch_size=128) for k,v in splits.items()}
                    final={k:evaluate(model,v) for k,v in loaders.items()}
                    ckpt=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(ckpt['model_state'])
                    best={k:evaluate(model,v) for k,v in loaders.items()}
                    exported=model.export_classical()
                    with torch.no_grad():
                        torch.testing.assert_close(model(x),exported(x),atol=1e-10,rtol=1e-9)
                        delta=float((model(x)-exported(x)).abs().max())
                    torch.save(exported.state_dict(),folder/'classical-export.pt')
                    write_json(folder/'result.json',dict(status='completed',condition=condition,seed=seed,
                        parameters=sum(p.numel() for p in model.parameters()),best=best,final=final,
                        best_epoch=ckpt['epoch'],export_delta=delta,train_seconds=sum(r['train_seconds'] for r in h)))
        protocol['status']='completed';write_json(root/'protocol.json',protocol)
    except BaseException as e:
        import traceback
        (root/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        protocol['status']='failed';protocol['error']=str(e);write_json(root/'protocol.json',protocol);raise
    finally:torch.set_num_threads(old)
    print(root,flush=True);return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output);protocol=json.loads((root/'protocol.json').read_text())
    records=[];rows=[]
    for seed in protocol['seeds']:
        for condition in protocol['conditions']:
            folder=root/f'{condition}-{seed}'
            r={k:json.loads((folder/f'{k}.json').read_text()) for k in ['config','environment','initial','updates','result']}
            result=r['result'];h=pd.read_csv(folder/'metrics.csv')
            if result['status']!='completed' or result['condition']!=condition or result['seed']!=seed or len(h)!=protocol['epochs']:raise ValueError('Incomplete/inconsistent pilot')
            r['metrics']=h.to_dict('records');records.append(r)
            rows.append(dict(condition=condition,seed=seed,parameters=result['parameters'],best_epoch=result['best_epoch'],
                train_accuracy=result['best']['train']['accuracy'],val_accuracy=result['best']['validation']['accuracy'],
                val_loss=result['best']['validation']['loss'],final_val_accuracy=result['final']['validation']['accuracy'],
                export_delta=result['export_delta']))
    output.mkdir(parents=True,exist_ok=True)
    f=pd.DataFrame(rows);f.to_csv(output/'runs.csv',index=False)
    write_json(output/'manifest.json',dict(protocol=protocol,runs=records))
    a=f.groupby('condition').agg(n=('seed','count'),train_mean=('train_accuracy','mean'),val_mean=('val_accuracy','mean'),val_std=('val_accuracy','std'),val_loss_mean=('val_loss','mean'))
    a.to_csv(output/'summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in records:
        h=pd.DataFrame(r['metrics'])
        for ax,key in zip(axes,['train_loss','val_accuracy']):
            ax.plot(h.epoch,h[key],label=f"{r['result']['condition']}/{r['result']['seed']}");ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# MNIST r2/r3機率後處理pilot\n\n'+('SMOKE，僅工程檢查。\n\n' if protocol['smoke'] else '')
    text+='Study: `'+root.name+'`\n\n'+a.fillna('N/A').to_markdown()+'\n\n'+f.to_markdown(index=False)
    text+='\n\n![curves](curves.png)\n\n兩組均457參數、global center；只改mapping的機率欄。Train/validation同best-validation-loss checkpoint，final另列。1000/200、50epochs不是論文完整資料；未使用official test。不把參數比當accuracy維持證據。公式對p做一次tanh/冪次，不再平方p。配置、來源假設、初始尺度/梯度與更新見manifest.json及docs/paper-r3-protocol.md。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(a.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--report')
    p.add_argument('--output',default='results');p.add_argument('--data',default='data/mnist');a=p.parse_args()
    if a.report:report(a.report,a.output)
    else:run(a.output,a.data,a.smoke)
