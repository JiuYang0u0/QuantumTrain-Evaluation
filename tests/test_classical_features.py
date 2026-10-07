import torch


def test_controls_match_mapping_and_feature_scale():
    from qt_eval.feature_controls import make_controls
    models=make_controls(11)
    reference=models['quantum_trainable']
    std=reference.feature_values().std(unbiased=False)
    for name,m in models.items():
        for p,q in zip(m.mapping.parameters(),reference.mapping.parameters()):
            torch.testing.assert_close(p,q)
        if name!='zero':
            torch.testing.assert_close(m.feature_values().std(unbiased=False),std)
    assert torch.count_nonzero(models['zero'].feature_values())==0
    assert models['classical_trainable'].coefficients.numel()==30
    assert not models['quantum_frozen'].q_params.requires_grad
    assert not models['random_fixed'].coefficients.requires_grad


def test_classical_feature_gradient_reaches_coefficients():
    from qt_eval.feature_controls import make_controls
    m=make_controls(22)['classical_trainable']
    x=torch.rand(20,64,dtype=torch.float64)
    loss=torch.nn.functional.cross_entropy(m(x),torch.arange(20)%10)
    loss.backward()
    assert torch.isfinite(m.coefficients.grad).all()
    assert m.coefficients.grad.norm()>0
