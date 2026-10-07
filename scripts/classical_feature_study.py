"""Five feature generators, paired initialization, train/validation only."""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from qt_eval.feature_controls import make_controls
from qt_eval.data import load_digits_splits
from qt_eval.experiment import evaluate
from qt_eval.artifacts import write_json,environment
from qt_eval.config import validate_config


def run(output,smoke=False):
    root=Path(output)/('classical-features-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    x,y=data['train'].tensors
    idx=torch.cat([torch.where(y==i)[0][:2] for i in range(10)]);x,y=x[idx],y[idx]
    protocol=dict(smoke=smoke,seeds=[11] if smoke else [11,22,33],epochs=2 if smoke else 200,
        conditions=['quantum_trainable','quantum_frozen','zero','random_fixed','classical_trainable'],
        learning_rate=.001,quantum_feature_scale=1000.,basis_seed=2026,basis_shape=[610,30],
        mapping='tanh',target='leaky_relu',weight_transform='center_weights',depth=3,q_delta=.001,
        classical_gain='match initial quantum feature std; fixed scalar thereafter',test_access=False,status='running')
    write_json(root/'protocol.json',protocol)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in ('train','validation')})
    write_json(root/'tiny_indices.json',[data['indices']['train'][i] for i in idx.tolist()])
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            models=make_controls(seed)
            for name,model in models.items():
                print(f'{name} seed={seed}',flush=True)
                folder=root/f'{name}-{seed}';folder.mkdir()
                config=validate_config(dict(seed=seed,epochs=protocol['epochs'],mapping_activation='tanh',
                    target_activation='leaky_relu',weight_transform='center_weights',feature_scale=1000.,mode='smoke' if smoke else 'full'))
                write_json(folder/'config.json',config|{'condition':name,'samples':20,'basis_seed':2026})
                write_json(folder/'environment.json',environment(config))
                write_json(folder/'result.json',{'status':'running'})
                try:
                    params=[p for p in model.parameters() if p.requires_grad]
                    optimizer=torch.optim.Adam(params,lr=.001)
                    loader=DataLoader(TensorDataset(x,y),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
                    validation=DataLoader(data['validation'],batch_size=64)
                    initial_feature=model.feature_values().detach().clone()
                    feature_param=model.q_params if hasattr(model,'q_params') else model.coefficients
                    initial_param=feature_param.detach().clone()
                    feature_trainable=feature_param.requires_grad
                    history=[];best=float('inf')
                    for epoch in range(1,protocol['epochs']+1):
                        model.train()
                        for bx,by in loader:
                            optimizer.zero_grad(set_to_none=True)
                            loss=F.cross_entropy(model(bx),by)
                            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                            loss.backward()
                            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                                raise RuntimeError('Missing/non-finite gradient')
                            grad=float(feature_param.grad.norm()) if feature_trainable else None
                            optimizer.step()
                        train=evaluate(model,DataLoader(TensorDataset(x,y),batch_size=64))
                        val=evaluate(model,validation)
                        with torch.no_grad():
                            f=model.feature_values()
                            z=torch.cat([model.codes,f[:,None]],dim=1);saturation=[]
                            for layer in model.mapping:
                                z=layer(z)
                                if isinstance(layer,torch.nn.Tanh):saturation.append(float((z.abs()>.99).double().mean()))
                            row=dict(epoch=epoch,train_accuracy=train['accuracy'],train_loss=train['loss'],
                                val_accuracy=val['accuracy'],val_loss=val['loss'],feature_grad_norm=grad,
                                feature_std=float(f.std(unbiased=False)),feature_delta=float((f-initial_feature).norm()),
                                feature_parameter_delta=float((feature_param-initial_param).abs().max()),tanh_saturation=saturation)
                        if not feature_trainable and row['feature_parameter_delta']!=0:
                            raise RuntimeError('Fixed feature parameters changed')
                        history.append(row);write_json(folder/'history.json',history)
                        if val['loss']<best:
                            best=val['loss'];torch.save(dict(model_state=model.state_dict(),epoch=epoch),folder/'best.pt')
                    checkpoint=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(checkpoint['model_state'])
                    result=dict(status='completed',condition=name,seed=seed,initial_feature_std=float(initial_feature.std(unbiased=False)),
                        feature_trainable_parameters=feature_param.numel() if feature_trainable else 0,
                        mapping_parameters=sum(p.numel() for p in model.mapping.parameters()),
                        max_train_accuracy=max(h['train_accuracy'] for h in history),final_train_accuracy=history[-1]['train_accuracy'],
                        final_train_loss=history[-1]['train_loss'],best_epoch=checkpoint['epoch'],best_validation=evaluate(model,validation))
                    write_json(folder/'result.json',result)
                except BaseException as exc:
                    import traceback
                    (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                    write_json(folder/'result.json',dict(status='failed',error=str(exc)));raise
        protocol['status']='completed';write_json(root/'protocol.json',protocol)
    finally:
        torch.set_num_threads(old_threads)
    print(root,flush=True);return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output);protocol=json.loads((root/'protocol.json').read_text())
    rows=[];records=[]
    for seed in protocol['seeds']:
        for name in protocol['conditions']:
            folder=root/f'{name}-{seed}'
            record={k:json.loads((folder/f'{k}.json').read_text()) for k in ('result','config','environment','history')}
            r=record['result']
            if r['status']!='completed' or r['condition']!=name or r['seed']!=seed:raise ValueError('Incomplete/inconsistent run')
            if name in ('zero','random_fixed','quantum_frozen') and any(h['feature_parameter_delta']!=0 for h in record['history']):
                raise ValueError('Fixed feature changed')
            records.append(record)
            rows.append({k:r[k] for k in ('condition','seed','max_train_accuracy','final_train_accuracy','final_train_loss','best_epoch','initial_feature_std','feature_trainable_parameters')}|
                        dict(val_accuracy=r['best_validation']['accuracy'],val_loss=r['best_validation']['loss']))
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'runs.csv',index=False)
    aggregate=frame.groupby('condition').agg(n=('seed','count'),max_train_mean=('max_train_accuracy','mean'),
        final_train_mean=('final_train_accuracy','mean'),final_train_std=('final_train_accuracy','std'),
        val_mean=('val_accuracy','mean'),val_std=('val_accuracy','std'),val_loss_mean=('val_loss','mean'))
    aggregate.to_csv(output/'summary.csv');write_json(output/'manifest.json',dict(protocol=protocol,runs=records))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in records:
        h=pd.DataFrame(r['history'])
        for ax,key in zip(axes,['train_accuracy','val_loss']):
            ax.plot(h.epoch,h[key],label=f"{r['result']['condition']}/{r['result']['seed']}");ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend(fontsize=5);fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# 量子與傳統特徵重訓對照\n\nStudy: `'+root.name+'`\n\n'+('SMOKE：僅工程檢查。\n\n' if protocol['smoke'] else '')
    text+=aggregate.fillna('N/A').to_markdown()+'\n\n'+frame.to_markdown(index=False)
    text+='\n\n![curves](curves.png)\n\n每線一個condition/seed，20筆train，未使用test。可訓練量子/傳統各30參數；相同mapping初始權重與初始非零feature標準差，但不同feature表示、梯度幾何和計算成本，不是完整等價比較。Random fixed與classical trainable起始feature逐元素相同，後者學習30維固定隨機基底係數；不是610維自由embedding。Gain只初始化校準一次。Best依validation loss選取；低表現小樣本不能支持一般量子優勢/無效結論。配置、環境、歷史见manifest.json。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(aggregate.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true')
    p.add_argument('--report');p.add_argument('--output',default='results');a=p.parse_args()
    if a.report:report(a.report,a.output)
    else:run(a.output,a.smoke)
