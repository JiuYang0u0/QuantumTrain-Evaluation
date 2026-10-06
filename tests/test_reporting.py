import json
import math
import pytest


def fixture_run(root,seed,accuracy,status='completed',**changes):
    from qt_eval.config import validate_config
    p=root/str(seed)
    p.mkdir()
    config=validate_config({'model':'classical','mode':'full','epochs':20,'seed':seed}|changes)
    (p/'config.json').write_text(json.dumps(config))
    result={'status':status,'test':{'accuracy':accuracy,'loss':1.0,'sample_count':360},
            'train_seconds':2.0,'parameters':{'target':610,'mapping':0,'quantum':0,'total':610}}
    (p/'result.json').write_text(json.dumps(result))
    return p


def test_summary_uses_sample_sd_and_identifies_missing_seeds(tmp_path):
    from qt_eval.reporting import summarize_runs
    runs=[fixture_run(tmp_path,11,.6),fixture_run(tmp_path,22,.8)]
    row=summarize_runs(runs).iloc[0]
    assert row['accuracy_mean']==pytest.approx(.7)
    assert row['accuracy_std']==pytest.approx(math.sqrt(.02))
    assert row['status']=='incomplete'
    assert row['missing_seeds']==[33]


def test_failed_and_single_seed_are_explicit(tmp_path):
    from qt_eval.reporting import summarize_runs
    runs=[fixture_run(tmp_path,11,.6),fixture_run(tmp_path,22,.8,'failed')]
    row=summarize_runs(runs).iloc[0]
    assert row['failed']==1
    assert math.isnan(row['accuracy_std'])
    assert row['status']=='incomplete'


def test_duplicate_and_incompatible_protocol_rejected(tmp_path):
    from qt_eval.reporting import summarize_runs
    a=fixture_run(tmp_path,11,.6)
    with pytest.raises(ValueError,match='Duplicate'):
        summarize_runs([a,a])
    b=fixture_run(tmp_path,22,.8,epochs=2)
    with pytest.raises(ValueError,match='protocol'):
        summarize_runs([a,b])


def test_smoke_requires_explicit_opt_in(tmp_path):
    from qt_eval.reporting import summarize_runs
    p=fixture_run(tmp_path,11,.6,mode='smoke')
    with pytest.raises(ValueError,match='Smoke'):
        summarize_runs([p])
    assert summarize_runs([p],allow_smoke=True).iloc[0]['expected']==1


def test_missing_entire_group_is_reported(tmp_path):
    from qt_eval.reporting import summarize_runs
    p=fixture_run(tmp_path,11,.6)
    frame=summarize_runs([p],expected_groups=['classical','quantum-depth1'])
    missing=frame[frame.group=='quantum-depth1'].iloc[0]
    assert missing['completed']==0 and missing['status']=='incomplete'
