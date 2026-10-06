"""Validation-selected training; final test is outside fit."""
import random
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .artifacts import write_json, environment
from .config import validate_config
from .data import load_digits_splits
from .models import make_model, TARGET_PARAMETERS
from .diagnostics import check_training_step


def evaluate(model, loader):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    with torch.no_grad():
        for x,y in loader:
            logits = model(x)
            loss = F.cross_entropy(logits,y,reduction='sum')
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite evaluation loss')
            loss_sum += float(loss)
            correct += int((logits.argmax(1)==y).sum())
            count += len(y)
    if count == 0:
        raise ValueError('Empty evaluation data')
    return dict(loss=loss_sum/count, accuracy=correct/count, sample_count=count)


def fit(model, splits, config, run_dir):
    run_dir = Path(run_dir)
    generator = torch.Generator().manual_seed(config['seed'])
    train = DataLoader(splits['train'],batch_size=config['batch_size'],shuffle=True,generator=generator)
    val = DataLoader(splits['validation'],batch_size=config['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    history, best = [], float('inf')
    for epoch in range(1,config['epochs']+1):
        model.train()
        loss_sum, count, train_seconds = 0.,0,0.
        for x,y in train:
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x),y)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad):
                raise RuntimeError('missing or non-finite training gradient')
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError('non-finite parameter after update')
            train_seconds += time.perf_counter()-start
            loss_sum += float(loss.detach())*len(y)
            count += len(y)
        metrics = evaluate(model,val)
        row = dict(epoch=epoch,train_loss=loss_sum/count,val_loss=metrics['loss'],
                   val_accuracy=metrics['accuracy'],train_seconds=train_seconds)
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir/'metrics.csv',index=False)
        if metrics['loss'] < best:
            best = metrics['loss']
            torch.save(dict(model_state=model.state_dict(),epoch=epoch,val_loss=best),run_dir/'best.pt')
    return history


def run_experiment(config, output_root):
    config = validate_config(config)
    root = Path(output_root)
    root.mkdir(parents=True,exist_ok=True)
    run_dir = root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:10])
    run_dir.mkdir()
    write_json(run_dir/'config.json',config)
    write_json(run_dir/'result.json',{'status':'running'})
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    random_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        torch.set_num_threads(config['torch_threads'])
        with torch.random.fork_rng(devices=[]):
            random.seed(config['seed'])
            np.random.seed(config['seed'])
            torch.manual_seed(config['seed'])
            write_json(run_dir/'environment.json',environment(config))
            splits = load_digits_splits(config['split_seed'],config['mode']=='smoke')
            write_json(run_dir/'split_indices.json',splits['indices'])
            model = make_model(config)
            x,y = splits['train'].tensors
            diagnostic = check_training_step(model,x[:32],y[:32])
            write_json(run_dir/'diagnostics.json',diagnostic)
            if diagnostic['status'] != 'passed':
                raise RuntimeError('Training diagnostic failed: '+diagnostic.get('reason','missing/zero/non-finite gradient or update'))
            history = fit(model,{k:splits[k] for k in ('train','validation')},config,run_dir)
            checkpoint = torch.load(run_dir/'best.pt',weights_only=True)
            model.load_state_dict(checkpoint['model_state'])
            test = evaluate(model,DataLoader(splits['test'],batch_size=config['batch_size']))
            parameters = dict(target=TARGET_PARAMETERS,
                              mapping=sum(p.numel() for p in model.mapping.parameters()) if hasattr(model,'mapping') else 0,
                              quantum=model.q_params.numel() if hasattr(model,'q_params') else 0,
                              total=sum(p.numel() for p in model.parameters()))
            write_json(run_dir/'result.json',dict(status='completed',best_epoch=checkpoint['epoch'],
                       test=test,train_seconds=sum(r['train_seconds'] for r in history),
                       end_to_end_seconds=time.perf_counter()-started,parameters=parameters))
    except BaseException as exc:
        (run_dir/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(run_dir/'result.json',dict(status='failed',error=str(exc)))
        raise
    finally:
        torch.set_num_threads(previous_threads)
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    return run_dir
