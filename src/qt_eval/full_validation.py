"""Full-training-set comparison; the runner has no test dataset interface."""
import time
import traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .artifacts import write_json,environment
from .config import validate_config
from .models import ClassicalMLP
from .feature_controls import make_controls
from .experiment import fit,evaluate

CONDITIONS=['direct_mlp','classical_trainable','quantum_trainable','quantum_frozen']


def models_for_seed(seed):
    controls=make_controls(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        direct=ClassicalMLP('leaky_relu').double()
    return {'direct_mlp':direct,**{k:controls[k] for k in CONDITIONS[1:]}}


def run_condition(model,condition,seed,splits,output,epochs=20,smoke=False):
    if set(splits)!={'train','validation'}:
        raise ValueError('Only train and validation splits accepted')
    if condition not in CONDITIONS:raise ValueError('Unknown condition')
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    if any(path.iterdir()):raise FileExistsError('Run directory must be empty')
    config=validate_config(dict(mode='smoke' if smoke else 'full',epochs=epochs,seed=seed,
        mapping_activation='tanh',target_activation='leaky_relu',weight_transform='center_weights',feature_scale=1000.))
    write_json(path/'config.json',config|{'condition':condition,'test_access':False,
        'weight_transform': 'none' if condition=='direct_mlp' else 'center_weights'})
    write_json(path/'result.json',{'status':'running'})
    old_threads=torch.get_num_threads();started=time.perf_counter()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            write_json(path/'environment.json',environment(config))
            initial_frozen={k:p.detach().clone() for k,p in model.named_parameters() if not p.requires_grad}
            history=fit(model,splits,config,path)
            frozen_delta=max((float((p.detach()-initial_frozen[k]).abs().max()) for k,p in model.named_parameters() if k in initial_frozen),default=0.)
            if frozen_delta!=0:raise RuntimeError('Frozen parameters changed')
            final_train=evaluate(model,DataLoader(splits['train'],batch_size=64))
            final_val=evaluate(model,DataLoader(splits['validation'],batch_size=64))
            checkpoint=torch.load(path/'best.pt',weights_only=True)
            model.load_state_dict(checkpoint['model_state'])
            result=dict(status='completed',condition=condition,seed=seed,
                train_samples=len(splits['train']),validation_samples=len(splits['validation']),
                final_train=final_train,final_validation=final_val,
                best_train=evaluate(model,DataLoader(splits['train'],batch_size=64)),
                best_validation=evaluate(model,DataLoader(splits['validation'],batch_size=64)),
                best_epoch=checkpoint['epoch'],frozen_parameter_delta=frozen_delta,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                parameter_elements=sum(p.numel() for p in model.parameters()),
                buffer_elements=sum(b.numel() for b in model.buffers()),
                train_seconds=sum(h['train_seconds'] for h in history),
                end_to_end_seconds=time.perf_counter()-started)
            write_json(path/'result.json',result);return result
    except BaseException as exc:
        (path/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(path/'result.json',dict(status='failed',error=str(exc)));raise
    finally:torch.set_num_threads(old_threads)
