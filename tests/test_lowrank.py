import torch
import pytest
from qt_eval.lowrank import LowRankCNN, parameters_for_rank, MAX_RANK
from qt_eval.paper_mnist import TargetCNN


def test_parameter_counts():
    assert parameters_for_rank(1) == 537
    assert parameters_for_rank(2) == 1024
    assert parameters_for_rank(3) == 1511
    assert parameters_for_rank(4) == 1998


def test_full_rank_reconstructs_target_exactly():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    target = TargetCNN().double()
    model = LowRankCNN(MAX_RANK, target=target)
    assert sum(p.numel() for p in model.parameters()) == parameters_for_rank(MAX_RANK)
    flat_target = torch.cat([p.flatten() for p in target.parameters()])
    torch.testing.assert_close(model.weight_tensor(), flat_target, atol=1e-8, rtol=1e-8)
    x = torch.randn(2, 1, 28, 28, dtype=torch.float64)
    torch.testing.assert_close(model(x), target(x), atol=1e-8, rtol=1e-8)


def test_forward_gradient_and_export_consistency():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = LowRankCNN(3)
    x = torch.randn(3, 1, 28, 28, dtype=torch.float64)
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(float(p.grad.norm()) > 0 for p in model.parameters())
    with torch.no_grad():
        torch.testing.assert_close(model(x), model.export_classical()(x), atol=1e-10, rtol=1e-9)


def test_rank_bounds():
    with pytest.raises(ValueError):
        LowRankCNN(0)
    with pytest.raises(ValueError):
        LowRankCNN(MAX_RANK + 1)
