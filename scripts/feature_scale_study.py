"""2x2 feature scale / quantum trainability study, no test access."""
import argparse
import json
import uuid
from pathlib import Path
import torch
import pandas as pd
from qt_eval.data import load_digits_splits
from qt_eval.artifacts import write_json
from qt_eval.quantum import QuantumTrain
from qt_eval.sensitivity import measure_sensitivity
from qt_eval.training_diagnostics import run_diagnostic


def run(output,smoke=False):
    root=Path(output)/('feature-scale-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={k:data[k] for k in ('train','validation')}
    x,y=splits['train'].tensors
    idx=torch.cat([torch.where(y==i)[0][:2] for i in range(10)]);x,y=x[idx],y[idx]
    protocol=dict(smoke=smoke,seeds=[11] if smoke else [11,22,33],epochs=2 if smoke else 200,
        scales=[1.,1000.],conditions=['trainable','frozen'],mapping='tanh',target='leaky_relu',
        weight_transform='center_weights',depth=3,learning_rate=.001,q_delta=.001,test_access=False,status='running')
    write_json(root/'protocol.json',protocol)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    write_json(root/'tiny_indices.json',[data['indices']['train'][i] for i in idx.tolist()])
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in protocol['seeds']:
            for scale in protocol['scales']:
                torch.manual_seed(seed)
                m=QuantumTrain(3,mapping_activation='tanh',target_activation='leaky_relu',weight_transform='center_weights',feature_scale=scale)
                write_json(root/f'initial-{seed}-{scale:g}.json',measure_sensitivity(m,x,y))
                for condition in protocol['conditions']:
                    print(f'scale={scale:g} {condition} seed={seed}',flush=True)
                    parent=root/f'scale-{scale:g}'/condition
                    run_diagnostic('tanh',seed,splits,parent,epochs=protocol['epochs'],tiny=True,
                        target_activation='leaky_relu',weight_transform='center_weights',
                        train_quantum=condition=='trainable',feature_scale=scale)
                    folder=parent/f'tanh-seed{seed}-tiny'
                    m.load_state_dict(torch.load(folder/'best.pt',weights_only=True)['model_state'])
                    write_json(folder/'sensitivity.json',measure_sensitivity(m,x,y))
        protocol['status']='completed';write_json(root/'protocol.json',protocol)
    finally:
        torch.set_num_threads(previous)
    print(root,flush=True)
    return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output)
    protocol=json.loads((root/'protocol.json').read_text())
    rows=[];records=[]
    for seed in protocol['seeds']:
        for scale in protocol['scales']:
            for condition in protocol['conditions']:
                folder=root/f'scale-{scale:g}'/condition/f'tanh-seed{seed}-tiny'
                r={k:json.loads((folder/f'{k}.json').read_text()) for k in ['config','environment','result','diagnostic','sensitivity']}
                if r['result']['status']!='completed' or r['config']['feature_scale']!=scale or r['config']['train_quantum']!=(condition=='trainable'):
                    raise ValueError('Incomplete/inconsistent run')
                d=r['diagnostic'];h=d['history']
                if condition=='frozen' and any(z['quantum_max_delta']!=0 for z in h):
                    raise ValueError('Frozen parameter changed')
                r['condition']=condition;records.append(r)
                rows.append(dict(scale=scale,condition=condition,seed=seed,
                    max_train_accuracy=max(z['train_accuracy'] for z in h),final_train_accuracy=h[-1]['train_accuracy'],
                    best_val_accuracy=d['best_validation']['accuracy'],best_val_loss=d['best_validation']['loss'],best_epoch=d['best_epoch'],
                    peak_saturation=max(max(z['observation']['mapping_tanh_saturation']) for z in h),
                    best_scaled_feature_std=r['sensitivity']['scaled_feature_std'],
                    zero_feature_weight_delta=r['sensitivity']['zero_feature_weight_relative_delta']))
    output.mkdir(parents=True,exist_ok=True)
    f=pd.DataFrame(rows);f.to_csv(output/'runs.csv',index=False)
    a=f.groupby(['scale','condition']).agg(n=('seed','count'),max_train_mean=('max_train_accuracy','mean'),
        final_train_mean=('final_train_accuracy','mean'),final_train_std=('final_train_accuracy','std'),
        val_mean=('best_val_accuracy','mean'),val_std=('best_val_accuracy','std'),val_loss_mean=('best_val_loss','mean'))
    a.to_csv(output/'summary.csv')
    write_json(output/'manifest.json',dict(protocol=protocol,runs=records,
        initial_sensitivity={p.stem:json.loads(p.read_text()) for p in root.glob('initial-*.json')}))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in records:
        h=pd.DataFrame(r['diagnostic']['history']);c=r['config']
        for ax,key in zip(axes,['train_accuracy','val_loss']):
            ax.plot(h.epoch,h[key],label=f"{c['feature_scale']:g}/{r['condition']}/{c['seed']}")
            ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend(fontsize=5);fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# 量子特徵尺度 × 可訓練性\n\nStudy: `'+root.name+'`\n\n'+('SMOKE，不作研究結論。\n\n' if protocol['smoke'] else '')
    text+=a.fillna('N/A').to_markdown()+'\n\n'+f.to_markdown(index=False)
    text+='\n\n![curves](curves.png)\n\n每線一個scale/condition/seed；20筆固定train。倍率作用於機率後處理與中心化之後、mapping之前，並改變量子梯度尺度。Frozen仍使用初始量子特徵。飽和比例為每層Tanh輸出abs>.99，peak_saturation取訓練各epoch/層最大值。Best由validation loss選取；未使用test。三seed僅描述統計，feature置零未重訓不是獨立baseline。配置、環境、診斷、敏感度見manifest.json。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(a.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true')
    p.add_argument('--report');p.add_argument('--output',default='results');a=p.parse_args()
    if a.report: report(a.report,a.output)
    else: run(a.output,a.smoke)
