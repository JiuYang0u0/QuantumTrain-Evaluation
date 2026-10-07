"""Fixed centered-weight control versus differentiable fan-in scaling."""
import argparse
import json
import uuid
from pathlib import Path
import torch
from qt_eval.artifacts import write_json
from qt_eval.data import load_digits_splits
from qt_eval.training_diagnostics import run_diagnostic
from centering_study import initialization_evidence,report as centered_report


def run(output,smoke=False):
    root=Path(output)/('fanin-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    data=load_digits_splits(smoke=smoke)
    splits={k:data[k] for k in ('train','validation')}
    x,y=splits['train'].tensors
    idx=torch.cat([torch.where(y==i)[0][:2] for i in range(10)])
    protocol=dict(smoke=smoke,seeds=[11] if smoke else [11,22,33],epochs=2 if smoke else 200,
        transforms=['center_weights','center_fanin'],mapping='tanh',target='leaky_relu',negative_slope=.01,
        depth=3,q_delta=.001,learning_rate=.001,train_samples=20,test_access=False,status='running',
        target_std_W1=1/(3*64)**.5,target_std_W2=1/(3*8)**.5,variance_floor=1e-12,
        definition='Per forward: center W1/W2, optionally divide by floored std and multiply target std; bias unchanged')
    write_json(root/'protocol.json',protocol)
    write_json(root/'split_indices.json',{k:data['indices'][k] for k in splits})
    write_json(root/'tiny_indices.json',[data['indices']['train'][i] for i in idx.tolist()])
    write_json(root/'initialization.json',initialization_evidence(x[idx]))
    for seed in protocol['seeds']:
        for transform in protocol['transforms']:
            print(f'{transform} seed={seed}',flush=True)
            run_diagnostic('tanh',seed,splits,root/transform,epochs=protocol['epochs'],tiny=True,
                           target_activation='leaky_relu',weight_transform=transform)
    protocol['status']='completed';write_json(root/'protocol.json',protocol)
    print(root,flush=True)
    return root


def report(root,output):
    centered_report(root,output)
    path=Path(output)/'study.md'
    text=path.read_text(encoding='utf-8')
    text=text.replace('# 生成權重中心化對照','# 中心化後的fan-in尺度對照')
    text=text.replace('中心化僅W1/W2各自減平均，不改bias、不除以std、不做fan-in scaling。',
        '控制組僅中心化W1/W2；處理組每次forward再除以floored std，乘以1/sqrt(3*fan_in)，bias保持不變。這是持續的可微分重參數化，不只是初始化；不是He initialization，也不產生獨立均勻權重。')
    path.write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true')
    p.add_argument('--report');p.add_argument('--output',default='results');a=p.parse_args()
    if a.report: report(a.report,a.output)
    else: run(a.output,a.smoke)
