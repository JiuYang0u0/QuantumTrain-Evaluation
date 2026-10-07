"""Controlled affine/Tanh study with a fixed LeakyReLU target; train/validation only."""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from qt_eval.artifacts import write_json
from qt_eval.data import load_digits_splits
from qt_eval.quantum import QuantumTrain
from qt_eval.training_diagnostics import run_diagnostic


def weight_structure(weights):
    matrix=weights[:512].reshape(8,64)
    additive=matrix.mean(0,keepdim=True)+matrix.mean(1,keepdim=True)-matrix.mean()
    return dict(singular_values=torch.linalg.svdvals(matrix).tolist(),
                nonadditive_relative=float((matrix-additive).norm()/matrix.norm().clamp_min(1e-15)))


def run(output,smoke=False):
    root=Path(output)/('leaky-mapping-'+uuid.uuid4().hex[:10])
    root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={key:data[key] for key in ('train','validation')}
    protocol=dict(target_activation='leaky_relu',negative_slope=.01,depth=3,q_delta=.001,
                  mappings=['affine','tanh'],seeds=[11] if smoke else [11,22,33],
                  epochs=2 if smoke else 200,train_samples=20,split_seed=2026,
                  optimizer='Adam',learning_rate=.001,smoke=smoke,test_access=False)
    write_json(root/'protocol.json',protocol)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    rows=[]
    for seed in protocol['seeds']:
        for mapping in protocol['mappings']:
            print(f'{mapping}, fixed LeakyReLU, seed={seed}',flush=True)
            result=run_diagnostic(mapping,seed,splits,root,epochs=protocol['epochs'],tiny=True,target_activation='leaky_relu')
            folder=root/f'{mapping}-seed{seed}-tiny'
            model=QuantumTrain(3,mapping_activation=mapping,target_activation='leaky_relu')
            model.load_state_dict(torch.load(folder/'best.pt',weights_only=True)['model_state'])
            with torch.no_grad():
                structure=weight_structure(model.generated_weights())
            write_json(folder/'structure.json',structure)
            rows.append(dict(mapping=mapping,seed=seed,best_epoch=result['best_epoch'],
                      maximum_train_accuracy=max(h['train_accuracy'] for h in result['history']),
                      final_train_accuracy=result['history'][-1]['train_accuracy'],
                      best_train_accuracy=result['best_train']['accuracy'],
                      best_val_accuracy=result['best_validation']['accuracy'],
                      nonadditive_relative=structure['nonadditive_relative']))
            pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    print(root)
    return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    records=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for seed in protocol['seeds']:
        for mapping in protocol['mappings']:
            folder=root/f'{mapping}-seed{seed}-tiny'
            record={'run_id':folder.name}
            for key in ('config','environment','result','diagnostic','structure'):
                record[key]=json.loads((folder/f'{key}.json').read_text())
            if record['result']['status']!='completed':
                raise ValueError('Incomplete run: '+folder.name)
            records.append(record)
            h=pd.DataFrame(record['diagnostic']['history'])
            axes[0].plot(h.epoch,h.train_accuracy,label=f'{mapping}-{seed}')
            axes[1].plot(h.epoch,h.val_loss,label=f'{mapping}-{seed}')
    for ax,label in zip(axes,['Train accuracy (20 examples)','Validation cross entropy']):
        ax.set(xlabel='Epoch',ylabel=label);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    frame=pd.read_csv(root/'summary.csv')
    frame.to_csv(output/'runs.csv',index=False)
    write_json(output/'manifest.json',dict(protocol=protocol,runs=records))
    text='# 固定 LeakyReLU 的 Mapping 對照\n\nStudy: `'+root.name+'`\n\n'
    text+=('SMOKE：不可用於研究結論。\n\n' if protocol['smoke'] else 'Train/validation-only 探索性實驗，未使用test。\n\n')
    text+=frame.to_markdown(index=False)+'\n\n![curves](curves.png)\n\n左：固定20筆train的擬合；右：validation loss。每條線一個seed。最高train accuracy是能力診斷；best checkpoint只由validation loss選取。權重nonadditive_relative是best checkpoint第一層8×64矩陣扣除row/column加性近似後的相對Frobenius norm，不是任務效能或量子優勢指標。\n'
    text+='\n## 凍結設定\n```json\n'+json.dumps(protocol,indent=2)+'\n```\n\n完整配置、環境、每epoch診斷和奇異值保存在manifest.json。三seeds只供描述；非線性改善不證明量子分支的增量價值。\n'
    (output/'study.md').write_text(text,encoding='utf-8')
    print(frame.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report')
    parser.add_argument('--output',default='results')
    args=parser.parse_args()
    if args.report: report(args.report,args.output)
    else: run(args.output,args.smoke)
