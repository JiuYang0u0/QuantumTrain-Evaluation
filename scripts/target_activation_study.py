"""Frozen 20-example ReLU/LeakyReLU experiment; no test-set evaluation."""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from qt_eval.data import load_digits_splits
from qt_eval.training_diagnostics import run_diagnostic
from qt_eval.artifacts import write_json


def run(output,smoke=False):
    root=Path(output)/('target-'+uuid.uuid4().hex[:10])
    root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={k:data[k] for k in ['train','validation']}
    protocol=dict(depth=3,mapping='affine',targets=['relu','leaky_relu'],negative_slope=.01,
                  seeds=[11] if smoke else [11,22,33],epochs=2 if smoke else 200,smoke=smoke,
                  train_samples=20,split_seed=2026,test_access=False)
    write_json(root/'protocol.json',protocol)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    rows=[]
    for seed in protocol['seeds']:
        for target in protocol['targets']:
            print(f'{target} seed={seed}',flush=True)
            d=run_diagnostic('affine',seed,splits,root/target,epochs=protocol['epochs'],tiny=True,target_activation=target)
            rows.append(dict(target=target,seed=seed,max_train_accuracy=max(h['train_accuracy'] for h in d['history']),
                             final_train_accuracy=d['history'][-1]['train_accuracy'],best_val_accuracy=d['best_validation']['accuracy'],
                             hidden_zero_fraction=d['final_observation']['hidden_zero_fraction'],
                             logits_std=d['final_observation']['logits_between_images_std']))
            pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    print(root)
    return root


def report(root,output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    rows=pd.read_csv(root/'summary.csv')
    assert len(rows)==len(protocol['seeds'])*2
    rows.to_csv(output/'runs.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for seed in protocol['seeds']:
        for target in protocol['targets']:
            p=root/target/f'affine-seed{seed}-tiny'
            d=json.loads((p/'diagnostic.json').read_text())
            h=pd.DataFrame(d['history'])
            axes[0].plot(h.epoch,h.train_accuracy,label=f'{target}-{seed}')
            axes[1].plot(h.epoch,[x['logits_between_images_std'] for x in h.observation],label=f'{target}-{seed}')
    for ax,label in zip(axes,['Train accuracy','Logits between-images std']):
        ax.set(xlabel='Epoch',ylabel=label);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# 目標 activation 對照\n\nStudy: '+root.name+'\n\n'+rows.to_markdown(index=False)
    text+='\n\n![curves](curves.png)\n\n左：固定20筆的訓練準確率；右：相同20筆的logits跨圖片標準差。每條線一個seed，未使用test。max_train_accuracy僅為擬合能力診斷，best_val_accuracy由最低validation loss checkpoint決定。\n\n協定：\n```json\n'+json.dumps(protocol,indent=2)+'\n```\n'
    (output/'target-study.md').write_text(text,encoding='utf-8')
    print(rows.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--report')
    p.add_argument('--output',default='results')
    args=p.parse_args()
    if args.report: report(args.report,args.output)
    else: run(args.output,args.smoke)
