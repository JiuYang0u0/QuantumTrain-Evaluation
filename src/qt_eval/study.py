"""Sequential matrix execution and validation-only cost pilot."""
import argparse
import time
import uuid
from pathlib import Path
import torch
from .config import validate_config, load_config
from .data import load_digits_splits
from .models import make_model
from .experiment import fit, run_experiment
from .artifacts import write_json
from .reporting import write_report

GROUPS=['classical','quantum-depth1','quantum-depth3','quantum-depth5']


def study_configs(base):
    base=validate_config(base)
    seeds=[11] if base['mode']=='smoke' else [11,22,33]
    return [validate_config(base|{'model':model,'q_depth':depth,'seed':seed})
            for model,depth in [('classical',3),('quantum',1),('quantum',3),('quantum',5)]
            for seed in seeds]


def pilot_training(base,output_root):
    config=validate_config(base|{'model':'quantum','q_depth':5,'epochs':1})
    path=Path(output_root)/('pilot-'+uuid.uuid4().hex[:10])
    path.mkdir(parents=True)
    previous_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(config['torch_threads'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'])
            data=load_digits_splits(config['split_seed'],config['mode']=='smoke')
            write_json(path/'config.json',config)
            history=fit(make_model(config),{k:data[k] for k in ('train','validation')},config,path)
            result={'train_seconds':history[0]['train_seconds'],'val_loss':history[0]['val_loss'],
                    'path':str(path),'purpose':'validation-only pilot; excluded from study'}
            write_json(path/'pilot.json',result)
            return result
    finally:
        torch.set_num_threads(previous_threads)


def run_study(base,output_root):
    base=validate_config(base)
    path=Path(output_root)/('study-'+uuid.uuid4().hex[:10])
    path.mkdir(parents=True)
    manifest={'base_config':base,'groups':GROUPS,'runs':[],'errors':[],'status':'running'}
    write_json(path/'study.json',manifest)
    for config in study_configs(base):
        before=set(path.iterdir())
        print(f"Running {config['model']} depth={config['q_depth']} seed={config['seed']}",flush=True)
        try:
            run=run_experiment(config,path)
            manifest['runs'].append(run.name)
        except Exception as error:
            created=[p for p in set(path.iterdir())-before if p.is_dir() and (p/'config.json').exists()]
            manifest['runs'].extend(p.name for p in created)
            manifest['errors'].append({'config':config,'error':str(error)})
        write_json(path/'study.json',manifest)
    manifest['status']='failed' if manifest['errors'] else 'completed'
    write_json(path/'study.json',manifest)
    if manifest['runs']:
        write_report([path/p for p in manifest['runs']],path/'report',base['mode']=='smoke',GROUPS)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/smoke.json')
    parser.add_argument('--output',default='results')
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    config=load_config(args.config)
    if args.pilot:
        print(pilot_training(config,args.output))
    else:
        path=run_study(config,args.output)
        print(path)
        import json
        if json.loads((path/'study.json').read_text())['status']!='completed':
            raise SystemExit(1)
