"""Read-only progress snapshots. Process presence never proves forward progress."""
import argparse
import csv
import json
import os
from datetime import datetime,timezone
from pathlib import Path


def read_json(path,errors):
    if not path.exists():return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        errors.append(f'{path.name}: {exc}');return {}


def process_snapshot():
    import psutil
    records=[];complete=True
    for process in psutil.process_iter(['pid','name']):
        try:
            if not any(k in (process.info['name'] or '').lower() for k in ('python','uv')):continue
            records.append(dict(pid=process.pid,cmdline=process.cmdline(),cwd=process.cwd()))
        except psutil.AccessDenied:complete=False
        except psutil.NoSuchProcess:pass
    return records,complete


def normalized(path):return os.path.normcase(str(Path(path).resolve()))


def nested(value,*keys):
    for key in keys:
        if not isinstance(value,dict):return None
        value=value.get(key)
    return value


def process_evidence(root,records,complete):
    project=root.parent.parent
    script='paper_depth_study.py' if root.name.startswith('paper-depth-') else (
        'paper_data_budget.py' if root.name.startswith(('paper-data-budget-','paper-data-extended-')) else None)
    confirmed=[];candidates=[]
    if script:
        for p in records:
            args=p['cmdline'];cwd=Path(p['cwd'])
            if not any(Path(a.replace('\\','/')).name==script for a in args):continue
            # --recover-from names an OLD source, never the newly created destination.
            if '--resume-root' in args:
                i=args.index('--resume-root')
                if i+1<len(args) and normalized(cwd/args[i+1])==normalized(root):confirmed.append(p['pid'])
                continue
            if any(a.startswith('--resume-root=') for a in args):
                value=next(a.split('=',1)[1] for a in args if a.startswith('--resume-root='))
                if normalized(cwd/value)==normalized(root):confirmed.append(p['pid'])
                continue
            if normalized(cwd)==normalized(project):candidates.append(p['pid'])
    evidence='confirmed' if confirmed else 'candidate' if candidates else 'not_found' if complete and script else 'unknown'
    return evidence,confirmed,candidates


def inspect_study(root,processes=None,visibility_complete=None,stale_seconds=600):
    root=Path(root).resolve()
    if not root.is_dir():raise ValueError(f'Study directory not found: {root}')
    if processes is None:processes,visibility_complete=process_snapshot()
    errors=[];protocol=read_json(root/'protocol.json',errors)
    if not isinstance(protocol,dict):raise ValueError('protocol.json must contain an object')
    state=read_json(root/'status.json',errors)
    study=read_json(root/'study.json',errors)
    if not isinstance(state,dict):errors.append('status.json is not an object');state={}
    if not isinstance(study,dict):errors.append('study.json is not an object');study={}
    expected=[];support='discovered_only'
    if root.name.startswith('paper-depth-') and 'seeds' in protocol:
        expected=[root/f"qt{protocol.get('blocks',106)}-seed{s}" for s in protocol['seeds']];support='declared_matrix'
    elif all(k in protocol for k in ('seeds','sizes','conditions')):
        expected=[root/f'{c}-n{n}-seed{s}' for s in protocol['seeds'] for n in protocol['sizes'] for c in protocol['conditions']];support='declared_matrix'
    elif isinstance(study,dict) and 'runs' in study:
        expected=[root/p for p in study['runs']]
    else:
        expected=sorted({p.parent for p in root.rglob('result.json')} | {p.parent for p in root.rglob('history.json')})
    evidence,pids,candidates=process_evidence(root,processes,bool(visibility_complete))
    now=datetime.now(timezone.utc);rows=[]
    for folder in expected:
        problems=[];result=read_json(folder/'result.json',problems)
        if not isinstance(result,dict):problems.append('result.json is not an object');result={}
        history=read_json(folder/'history.json',problems)
        if not history and (folder/'metrics.csv').exists():
            try:
                with (folder/'metrics.csv').open(encoding='utf-8',newline='') as f:history=list(csv.DictReader(f))
            except (OSError,ValueError) as exc:problems.append(str(exc))
        if not isinstance(history,list):history=[]
        last=history[-1] if history and isinstance(history[-1],dict) else {}
        completed=result.get('status')=='completed'
        unit='updates' if 'updates' in protocol or 'update' in last else 'epochs'
        step=last.get('update',last.get('epoch',0))
        if completed:step=result.get('updates',protocol.get('epochs',step))
        try:
            step=int(float(step))
            if step<0:raise ValueError('negative step')
        except (TypeError,ValueError,OverflowError):
            problems.append('Invalid saved step');step=0
        artifacts=[folder/n for n in ('history.json','metrics.csv','result.json','latest.pt') if (folder/n).exists()]
        modified=max((p.stat().st_mtime for p in artifacts),default=None)
        age=max(0.,now.timestamp()-modified) if modified is not None else None
        status=('completed_reported' if completed else 'failed_reported' if result.get('status')=='failed' or (folder/'error.txt').exists()
                else 'not_started' if not artifacts else 'process_present' if evidence=='confirmed'
                else 'interrupted_or_unowned' if evidence=='not_found' else 'unconfirmed')
        rows.append(dict(run=folder.name,status=status,saved_step=step,total=protocol.get('updates',protocol.get('epochs')),
            unit=unit,last_write_utc=datetime.fromtimestamp(modified,timezone.utc).isoformat() if modified is not None else None,
            age_seconds=round(age,1) if age is not None else None,stale=age is not None and age>stale_seconds,
            best_val_accuracy=nested(result,'best','validation','accuracy'),
            final_val_accuracy=nested(result,'final','validation','accuracy'),
            latest_val_accuracy=nested(last,'validation','accuracy') if isinstance(last.get('validation'),dict) else last.get('val_accuracy'),
            latest_checkpoint_exists=(folder/'latest.pt').exists(),best_checkpoint_exists=(folder/'best.pt').exists(),read_errors=problems))
    return dict(study=str(root),checked_at_utc=now.isoformat(),declared_status=state.get('status',protocol.get('status',study.get('status'))),
        matrix_support=support,completed=sum(r['status']=='completed_reported' for r in rows),expected=len(rows),
        process_evidence=evidence,confirmed_pids=pids,candidate_pids=candidates,visibility_complete=visibility_complete,
        stale_seconds=stale_seconds,read_errors=errors,runs=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',nargs='?',type=Path)
    parser.add_argument('--list',action='store_true',help='List study directories, not runtime status')
    parser.add_argument('--root',type=Path,default=Path('results'))
    parser.add_argument('--json',action='store_true');parser.add_argument('--stale-seconds',type=float,default=600)
    args=parser.parse_args()
    if args.list:
        paths=sorted({p.parent for name in ('protocol.json','study.json') for p in args.root.glob('*/'+name)})
        for p in paths:print(p)
        return
    if args.study is None:parser.error('Specify a study directory or --list')
    if args.stale_seconds<=0:parser.error('--stale-seconds must be positive')
    report=inspect_study(args.study,stale_seconds=args.stale_seconds)
    if args.json:print(json.dumps(report,ensure_ascii=False,indent=2));return
    print(report['study'])
    print(f"Saved complete: {report['completed']}/{report['expected']} | process evidence: {report['process_evidence']} | matrix: {report['matrix_support']}")
    print('Confirmed PIDs:',report['confirmed_pids'],'Candidate PIDs:',report['candidate_pids'])
    for r in report['runs']:
        print(f"{r['run']}: {r['status']} {r['saved_step']}/{r['total']} {r['unit']} | age={r['age_seconds']}s stale={r['stale']} | latest.pt={r['latest_checkpoint_exists']}")
        print('  latest/best/final validation:',r['latest_val_accuracy'],r['best_val_accuracy'],r['final_val_accuracy'])
        for error in r['read_errors']:print('  READ ERROR:',error)
    for error in report['read_errors']:print('READ ERROR:',error)
    print('Process presence is not a heartbeat. Saved steps may lag. Checkpoint existence is not validity verification.')


if __name__=='__main__':main()
