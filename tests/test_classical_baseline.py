import inspect
import math
import torch
import pytest
from qt_eval.classical_controls import ClassicalRandomFeatureTrain, build_mapping
from qt_eval.paper_mnist import PaperQuantumTrain, TargetCNN, unpack, functional_cnn


def test_mapping_matches_quantum_and_has_one_tanh():
    mapping = build_mapping()
    assert sum(p.numel() for p in mapping.parameters()) == 249
    assert sum(isinstance(layer, torch.nn.Tanh) for layer in mapping) == 1
    quantum = PaperQuantumTrain(16, global_center=True, probability_processing='upstream')
    assert sum(p.numel() for p in quantum.mapping.parameters()) == 249


def test_matched_parameter_counts_and_gradient():
    torch.set_num_threads(1)
    for latent, expected in [(208, 457), (1378, 1627)]:
        model = ClassicalRandomFeatureTrain(latent)
        total = sum(p.numel() for p in model.parameters())
        assert total == expected, (latent, total)
        assert model.z.numel() == latent
        x = torch.randn(2, 1, 28, 28, dtype=torch.float64)
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1]))
        loss.backward()
        assert model.z.grad is not None and torch.isfinite(model.z.grad).all()
        assert model.z.grad.norm() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.mapping.parameters())


def test_softmax_probabilities_and_export_consistency():
    torch.set_num_threads(1)
    model = ClassicalRandomFeatureTrain(208)
    p = model.probabilities()
    assert p.shape == (2 ** 13,)
    assert torch.all(p >= 0)
    assert float(p.sum().detach()) == pytest.approx(1.0, abs=1e-12)
    x = torch.randn(3, 1, 28, 28, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model.export_classical()(x), atol=1e-10, rtol=1e-9)
    assert model.projection.requires_grad is False


def test_init_nonquantum_is_deterministic_in_z_only():
    torch.manual_seed(11)
    a = ClassicalRandomFeatureTrain(208)
    torch.manual_seed(11)
    b = ClassicalRandomFeatureTrain(208)
    torch.testing.assert_close(a.z, b.z)
    torch.testing.assert_close(a.projection, b.projection)
