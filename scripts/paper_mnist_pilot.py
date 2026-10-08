"""Short real-MNIST pilot of explicit paper-guided reconstruction assumptions."""
import argparse
import time
import uuid
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from qt_eval.paper_mnist import TargetCNN,PaperQuantumTrain
from qt_eval.mnist_data import load_mnist_pilot
from qt_eval.config import validate_config
from qt_eval.artifacts import write_json,environment
from qt_eval.experiment import fit,evaluate


def run(output,data_root):
    root=Path(output)/('paper-mnist-pilot-'+uuid.uuid4().hex[:10]);root.mkdir(parents=True)
    protocol=dict(evidence='paper-guided reconstruction pilot, not exact reproduction',
        paper='arXiv:2405.11304v2',assumptions='docs/paper-mnist-protocol.md',seed=11,
        train_samples=1000,validation_samples=200,epochs=2,batch_size=128,lr=.0001,
        target_parameters=6690,qt_parameters=457,qubits=13,blocks=16,official_test_access=False,status='running')
    write_json(root/'protocol.json',protocol)
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        data=load_mnist_pilot(data_root)
        write_json(root/'split_indices.json',data['indices']);write_json(root/'dataset_hashes.json',data['hashes'])
        splits={k:data[k] for k in ('train','validation')}
        config=validate_config(dict(seed=11,epochs=2,batch_size=128,learning_rate=.0001,mode='full'))
        write_json(root/'environment.json',environment(config))
        for name in ['classical_cnn','qt16']:
            print(name,flush=True)
            folder=root/name;folder.mkdir()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(11)
                model=TargetCNN().double() if name=='classical_cnn' else PaperQuantumTrain()
                x,y=next(iter(DataLoader(splits['train'],batch_size=128)))
                start=time.perf_counter()
                loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward()
                grad_ok=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if not grad_ok:raise RuntimeError('Pilot gradient check failed')
                step=time.perf_counter()-start
                gradient=dict(all_finite=grad_ok,quantum_norm=float(model.theta.grad.norm()) if name=='qt16' else None,
                              forward_backward_seconds=step)
                model.zero_grad(set_to_none=True)
                history=fit(model,splits,config,folder)
                checkpoint=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(checkpoint['model_state'])
                val=evaluate(model,DataLoader(splits['validation'],batch_size=128))
                train=evaluate(model,DataLoader(splits['train'],batch_size=128))
                delta=None
                if name=='qt16':
                    exported=model.export_classical()
                    with torch.no_grad():
                        delta=float((model(x)-exported(x)).abs().max())
                    torch.testing.assert_close(model(x),exported(x),atol=1e-10,rtol=1e-9)
                    torch.save(exported.state_dict(),folder/'classical-export.pt')
                seconds=sum(h['train_seconds'] for h in history)
                result=dict(status='completed',evidence=protocol['evidence'],train=train,validation=val,
                    best_epoch=checkpoint['epoch'],trainable_parameters=sum(p.numel() for p in model.parameters()),
                    target_parameters=6690,parameter_ratio=sum(p.numel() for p in model.parameters())/6690,
                    gradient=gradient,export_max_logit_delta=delta,train_seconds=seconds,
                    estimated_50epoch_train_hours=seconds/(2*8)*(50*469)/3600)
                write_json(folder/'result.json',result);print(result,flush=True)
        protocol['status']='completed';write_json(root/'protocol.json',protocol)
    except BaseException as e:
        import traceback
        (root/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        protocol['status']='failed';protocol['error']=str(e);write_json(root/'protocol.json',protocol);raise
    finally:torch.set_num_threads(old)
    print(root,flush=True);return root


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results')
    p.add_argument('--data',default='data/mnist');a=p.parse_args();run(a.output,a.data)
