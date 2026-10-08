"""r1/r2 paired pilot; fixed assumptions apart from upstream global centering."""
import json
import uuid
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from qt_eval.paper_mnist import PaperQuantumTrain,TargetCNN
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.config import validate_config
from qt_eval.artifacts import write_json,environment
from qt_eval.experiment import fit,evaluate


def run(output='results',data_root='data/mnist'):
    root=Path(output)/('paper-centering-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    protocol=dict(evidence='paper-guided r2 diagnostic pilot, not reproduction',seeds=[11,22,33],
        conditions=['classical_cnn','r1_raw','r2_global_center'],train_samples=1000,validation_samples=200,
        epochs=2,batch_size=128,learning_rate=.0001,blocks=16,qt_parameters=457,target_parameters=6690,
        test_access=False,assumptions='docs/paper-mnist-protocol.md plus docs/paper-centering-protocol.md',status='running')
    write_json(root/'protocol.json',protocol)
    data=load_mnist_pilot(data_root);splits={k:data[k] for k in ('train','validation')}
    write_json(root/'split_indices.json',data['indices']);write_json(root/'dataset_hashes.json',data['hashes'])
    old=torch.get_num_threads();torch.set_num_threads(1);rows=[]
    try:
        for seed in protocol['seeds']:
            for condition in protocol['conditions']:
                print(condition,seed,flush=True)
                folder=root/f'{condition}-{seed}';folder.mkdir()
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model=TargetCNN().double() if condition=='classical_cnn' else PaperQuantumTrain(global_center=condition=='r2_global_center')
                    config=validate_config(dict(seed=seed,epochs=2,batch_size=128,learning_rate=.0001))
                    write_json(folder/'environment.json',environment(config))
                    write_json(folder/'config.json',protocol|{'seed':seed,'condition':condition})
                    history=fit(model,splits,config,folder)
                    checkpoint=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(checkpoint['model_state'])
                    train=evaluate(model,DataLoader(splits['train'],batch_size=128));val=evaluate(model,DataLoader(splits['validation'],batch_size=128))
                    export_delta=None
                    if condition!='classical_cnn':
                        x=splits['train'].tensors[0][:16]
                        exported=model.export_classical()
                        with torch.no_grad():
                            export_delta=float((model(x)-exported(x)).abs().max())
                            torch.testing.assert_close(model(x),exported(x),atol=1e-10,rtol=1e-9)
                        torch.save(exported.state_dict(),folder/'classical-export.pt')
                    result=dict(status='completed',condition=condition,seed=seed,train=train,validation=val,
                        parameters=sum(p.numel() for p in model.parameters()),best_epoch=checkpoint['epoch'],
                        export_delta=export_delta,train_seconds=sum(h['train_seconds'] for h in history))
                    write_json(folder/'result.json',result)
                    rows.append(dict(condition=condition,seed=seed,parameters=result['parameters'],train_accuracy=train['accuracy'],
                        val_accuracy=val['accuracy'],val_loss=val['loss'],export_delta=export_delta))
                    pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
        protocol['status']='completed';write_json(root/'protocol.json',protocol)
    except BaseException as e:
        import traceback
        (root/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        protocol['status']='failed';protocol['error']=str(e);write_json(root/'protocol.json',protocol);raise
    finally:torch.set_num_threads(old)
    print(pd.DataFrame(rows).to_string(index=False));print(root,flush=True)


if __name__=='__main__':run()
