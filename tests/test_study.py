import json


def test_matrix_and_manifest(tmp_path):
    from qt_eval.study import study_configs, run_study
    from qt_eval.config import validate_config
    assert len(study_configs(validate_config({'mode':'full','epochs':20})))==12
    config=validate_config({})
    assert len(study_configs(config))==4
    path=run_study(config,tmp_path)
    manifest=json.loads((path/'study.json').read_text())
    assert len(manifest['runs'])==4
    assert (path/'report'/'depth-study.md').exists()
    assert all(json.loads((path/p/'result.json').read_text())['status']=='completed' for p in manifest['runs'])


def test_pilot_is_validation_only(tmp_path,monkeypatch):
    import qt_eval.study as study
    from qt_eval.config import validate_config
    original=study.fit
    def checked(model,splits,config,path):
        assert set(splits)=={'train','validation'}
        return original(model,splits,config,path)
    monkeypatch.setattr(study,'fit',checked)
    result=study.pilot_training(validate_config({}),tmp_path)
    assert result['train_seconds']>0
    assert 'test' not in result
