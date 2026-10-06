"""Isolated one-step gradient and parameter-update health checks."""
import copy
import torch
from torch.nn import functional as F


def check_training_step(model, inputs, targets):
    with torch.random.fork_rng(devices=[]):
        clone = copy.deepcopy(model)
        clone.train()
        optimizer = torch.optim.Adam(clone.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clone(inputs), targets)
        if not torch.isfinite(loss):
            return {'status':'failed', 'reason':'non-finite loss', 'groups':{}}
        loss.backward()
        before = {name:p.detach().clone() for name,p in clone.named_parameters()}
        groups = {}
        for name,p in clone.named_parameters():
            key = 'quantum' if name == 'q_params' else ('mapping' if hasattr(clone,'q_params') else 'classical')
            groups.setdefault(key, []).append((name,p))
        details = {}
        for key, parameters in groups.items():
            present = all(p.grad is not None for _,p in parameters)
            finite = present and all(bool(torch.isfinite(p.grad).all()) for _,p in parameters)
            norm = sum(float(p.grad.norm())**2 for _,p in parameters if p.grad is not None)**0.5
            details[key] = dict(has_grad=present, finite=finite, norm=norm)
        if all(d['finite'] for d in details.values()):
            optimizer.step()
        for key, parameters in groups.items():
            details[key]['max_delta'] = max(float((p.detach()-before[name]).abs().max()) for name,p in parameters)
        passed = all(d['finite'] and d['norm'] > 0 and d['max_delta'] > 0 for d in details.values())
        return dict(status='passed' if passed else 'failed', loss=float(loss.detach()), groups=details)
