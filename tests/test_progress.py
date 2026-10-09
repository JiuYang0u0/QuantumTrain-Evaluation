import json
from pathlib import Path


def study(tmp_path):
    root=tmp_path/'paper-depth-demo';root.mkdir()
    (root/'protocol.json').write_text(json.dumps({'seeds':[11,22,33],'blocks':106,'updates':12000}))
    d=root/'qt106-seed11';d.mkdir()
    (d/'history.json').write_text(json.dumps([{'update':100,'validation':{'accuracy':.1}}]))
    return root


def test_running_file_without_process_is_not_live(tmp_path):
    from qt_eval.progress import inspect_study
    root=study(tmp_path)
    (root/'status.json').write_text('{"status":"running"}')
    r=inspect_study(root,processes=[],visibility_complete=True)
    assert r['process_evidence']=='not_found'
    assert r['runs'][0]['status']=='interrupted_or_unowned'
    assert r['runs'][0]['saved_step']==100
    assert len(r['runs'])==3


def test_explicit_resume_matches_only_correct_root(tmp_path):
    from qt_eval.progress import inspect_study
    root=study(tmp_path)
    p={'pid':123,'cmdline':['python','scripts/paper_depth_study.py','--resume-root',str(root)],'cwd':str(tmp_path)}
    r=inspect_study(root,processes=[p],visibility_complete=True)
    assert r['process_evidence']=='confirmed'
    assert r['runs'][0]['status']=='process_present'
    assert r['runs'][1]['status']=='not_started'
    p['cmdline'][-1]=str(tmp_path/'other')
    assert inspect_study(root,processes=[p],visibility_complete=True)['process_evidence']=='not_found'


def test_generic_process_is_candidate_not_confirmed(tmp_path):
    from qt_eval.progress import inspect_study
    root=study(tmp_path)
    project=root.parent.parent
    p={'pid':1,'cmdline':['python','scripts/paper_depth_study.py'],'cwd':str(project)}
    r=inspect_study(root,processes=[p],visibility_complete=True)
    assert r['process_evidence']=='candidate'
    assert r['runs'][0]['status']=='unconfirmed'


def test_partial_json_and_access_denied_are_not_completed(tmp_path):
    from qt_eval.progress import inspect_study
    root=study(tmp_path)
    (root/'qt106-seed11/result.json').write_text('{')
    r=inspect_study(root,processes=[],visibility_complete=False)
    assert r['process_evidence']=='unknown'
    assert r['runs'][0]['read_errors']
    assert r['completed']==0


def test_completed_is_reported_not_process_claim_and_checkpoint_not_loaded(tmp_path):
    from qt_eval.progress import inspect_study
    root=study(tmp_path);d=root/'qt106-seed11'
    (d/'result.json').write_text(json.dumps({'status':'completed','updates':12000,'best_update':11800,
        'best':{'validation':{'accuracy':.8}},'final':{'validation':{'accuracy':.79}}}))
    (d/'latest.pt').write_bytes(b'not a trusted checkpoint')
    r=inspect_study(root,processes=[],visibility_complete=True)
    row=r['runs'][0]
    assert row['status']=='completed_reported'
    assert row['saved_step']==12000 and row['latest_checkpoint_exists']
    assert row['best_val_accuracy']==.8 and row['final_val_accuracy']==.79


def test_malformed_history_row_does_not_crash_snapshot(tmp_path):
    from qt_eval.progress import inspect_study
    root=study(tmp_path)
    (root/'qt106-seed11/history.json').write_text('[{"update":"invalid","validation":null}]')
    (root/'status.json').write_text('[]')
    r=inspect_study(root,processes=[],visibility_complete=True)
    assert r['runs'][0]['read_errors']
    assert r['runs'][0]['saved_step']==0
