"""Frozen stage-two train/validation matrix and self-contained report."""
import argparse
import json
import uuid
from pathlib import Path
import pandas as pd
from qt_eval.artifacts import write_json
from qt_eval.data import load_digits_splits
from qt_eval.full_validation import CONDITIONS,models_for_seed,run_condition


def run(output,smoke=False):
    root=Path(output)/('full-validation-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={k:data[k] for k in ('train','validation')}
    p=dict(smoke=smoke,seeds=[11] if smoke else [11,22,33],epochs=2 if smoke else 20,
        conditions=CONDITIONS,batch_size=64,learning_rate=.001,split_seed=2026,
        target='64-8-10 LeakyReLU(.01)',mapping='11-40-200-40-1 Tanh',
        generator_weight_transform='center_weights',quantum_feature_scale=1000.,depth=3,q_delta=.001,
        test_access=False,train_samples=len(splits['train']),validation_samples=len(splits['validation']),status='running')
    write_json(root/'protocol.json',p)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    for seed in p['seeds']:
        models=models_for_seed(seed)
        for condition in p['conditions']:
            print(f'{condition} seed={seed}',flush=True)
            run_condition(models[condition],condition,seed,splits,root/f'{condition}-{seed}',p['epochs'],smoke)
    p['status']='completed';write_json(root/'protocol.json',p)
    print(root,flush=True);return root


def report(root,output):
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output)
    p=json.loads((root/'protocol.json').read_text());rows=[];records=[]
    for seed in p['seeds']:
        for condition in p['conditions']:
            folder=root/f'{condition}-{seed}'
            record={k:json.loads((folder/f'{k}.json').read_text()) for k in ['config','environment','result']}
            r=record['result']
            if r['status']!='completed' or r['condition']!=condition or r['seed']!=seed or r['frozen_parameter_delta']!=0:
                raise ValueError('Incomplete/inconsistent study')
            history=pd.read_csv(folder/'metrics.csv')
            if len(history)!=p['epochs']:raise ValueError('Incomplete epoch history')
            record['history']=history.to_dict('records');records.append(record)
            rows.append(dict(condition=condition,seed=seed,best_epoch=r['best_epoch'],
                best_train_accuracy=r['best_train']['accuracy'],best_val_accuracy=r['best_validation']['accuracy'],
                best_val_loss=r['best_validation']['loss'],final_train_accuracy=r['final_train']['accuracy'],
                final_val_accuracy=r['final_validation']['accuracy'],train_seconds=r['train_seconds'],
                trainable_parameters=r['trainable_parameters'],buffer_elements=r['buffer_elements']))
    output.mkdir(parents=True,exist_ok=True)
    f=pd.DataFrame(rows);f.to_csv(output/'runs.csv',index=False)
    a=f.groupby('condition').agg(n=('seed','count'),best_train_mean=('best_train_accuracy','mean'),
        val_mean=('best_val_accuracy','mean'),val_std=('best_val_accuracy','std'),val_loss_mean=('best_val_loss','mean'),
        final_train_mean=('final_train_accuracy','mean'),final_val_mean=('final_val_accuracy','mean'),train_seconds_mean=('train_seconds','mean'))
    a.to_csv(output/'summary.csv');write_json(output/'manifest.json',dict(protocol=p,runs=records))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in records:
        h=pd.DataFrame(r['history'])
        for ax,key in zip(axes,['train_loss','val_loss']):
            ax.plot(h.epoch,h[key],label=f"{r['result']['condition']}/{r['result']['seed']}");ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'curves.png',dpi=130);plt.close(fig)
    text='# 第二階段：完整train/validation比較\n\nStudy: `'+root.name+'`\n\n'+('SMOKE工程檢查。\n\n' if p['smoke'] else '')
    text+=a.fillna('N/A').to_markdown()+'\n\n'+f.to_markdown(index=False)
    text+='\n\n![curves](curves.png)\n\n每線一個condition/seed。Train loss是訓練過程中的樣本加權平均，validation loss是epoch結束後評估。Best train/validation均在同一最低validation loss checkpoint重測；final指最後epoch。沒有test評估；已用於選架構與checkpoint的validation不是獨立泛化證據。同epochs不等於同運算預算。直接MLP的參數化/初始化與生成器不同；傳統生成器有額外固定隨機基底儲存。Frozen量子仍在模擬器計算機率，時間未做快取最佳化。CPU計時僅描述當機器，不代表QPU或硬體優勢。詳見manifest協定及環境。\n'
    (output/'study.md').write_text(text,encoding='utf-8');print(a.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true')
    p.add_argument('--report');p.add_argument('--output',default='results');a=p.parse_args()
    if a.report:report(a.report,a.output)
    else:run(a.output,a.smoke)
