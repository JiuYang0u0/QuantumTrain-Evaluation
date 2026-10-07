"""Descriptive interventions, not quantum advantage or training improvements."""
import copy
import torch
from torch.nn import functional as F
from .quantum import probability_features,center_weight_blocks,fanin_weight_blocks
from .models import functional_mlp


def measure_sensitivity(model,x,y):
    with torch.random.fork_rng(devices=[]):
        m=copy.deepcopy(model)
        m.q_params.requires_grad_(True)
        loss=F.cross_entropy(m(x),y)
        grad=torch.autograd.grad(loss,m.q_params)[0]
        generator=torch.Generator().manual_seed(2026)
        direction=torch.randn(m.q_params.shape,generator=generator,dtype=m.q_params.dtype)
        direction=direction/direction.norm()
        with torch.no_grad():
            theta=m.q_params.clone()
            p=m.qnode(theta)
            w=m.generated_weights()
            logits=m(x)
            rows=[]
            for epsilon in [.001,.01,.1]:
                m.q_params.copy_(theta+epsilon*direction)
                pp=m.qnode(m.q_params);ww=m.generated_weights();ll=m(x)
                rows.append(dict(epsilon=epsilon,probability_l2_delta=float((pp-p).norm()),
                    weight_l2_delta=float((ww-w).norm()),weight_relative_delta=float((ww-w).norm()/w.norm().clamp_min(1e-15)),
                    logits_l2_delta=float((ll-logits).norm()),logits_relative_delta=float((ll-logits).norm()/logits.norm().clamp_min(1e-15)),
                    loss_delta=float(F.cross_entropy(ll,y)-loss)))
            m.q_params.copy_(theta)
            features=probability_features(p[:610],10)
            without=m.mapping(torch.cat([m.codes,torch.zeros(610,1,dtype=w.dtype)],dim=1)).flatten()
            if m.weight_transform=='center_weights':
                without=center_weight_blocks(without)
            elif m.weight_transform=='center_fanin':
                without=fanin_weight_blocks(without)
            zero_logits=functional_mlp(x,without,m.target_activation)
            return dict(quantum_loss_grad_norm=float(grad.norm()),feature_std=float(features.std(unbiased=False)),
                scaled_feature_std=float(features.std(unbiased=False))*m.feature_scale,
                perturbations=rows,zero_feature_weight_relative_delta=float((without-w).norm()/w.norm().clamp_min(1e-15)),
                zero_feature_logits_relative_delta=float((zero_logits-logits).norm()/logits.norm().clamp_min(1e-15)),
                zero_feature_loss_delta=float(F.cross_entropy(zero_logits,y)-loss),
                scope='one fixed normalized direction; zero-feature intervention without retraining; train probe only')
