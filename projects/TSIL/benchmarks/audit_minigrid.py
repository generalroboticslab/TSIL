"""Read-only audit of immutable banks, control identity and evaluation scope."""
import argparse,csv,hashlib,json
from pathlib import Path
from projects.TSIL.benchmarks.configurations import describe,fingerprint
from projects.TSIL.benchmarks.selection_configurations import covered,training_envelope
from projects.TSIL.benchmarks.suites import get_tsil_minigrid_suite


def control_identity(env,family):
    cells=[((x,y),env.grid.get(x,y)) for x in range(env.width) for y in range(env.height)
           if env.grid.get(x,y) is not None]
    targets=[]
    if family in ('GoToDoor','GoToObject'):targets=[list(map(int,env.target_pos))]
    elif family=='Fetch':targets=[list(p) for p,o in cells if o.type==env.targetType and o.color==env.targetColor]
    elif family in ('UnlockPickup','BlockedUnlockPickup','KeyCorridor'):targets=[list(p) for p,o in cells if o is env.obj]
    elif family=='RedBlueDoors':targets=[list(p) for p,o in cells if o is env.red_door]+[list(p) for p,o in cells if o is env.blue_door]
    matches=[[list(p),list(q)] for p,k in cells if k.type=='key' for q,d in cells
             if d.type=='door' and d.is_locked and k.color==d.color]
    return json.dumps(dict(grid=env.grid.encode()[:,:,[0,2]].tolist(),
        pos=list(map(int,env.agent_pos)),heading=int(env.agent_dir),targets=targets,matches=matches),sort_keys=True)


def audit(path):
    import gymnasium as gym
    bank=json.loads(path.read_text());suite=get_tsil_minigrid_suite(bank['suite']);rows=[]
    for task,data in bank['tasks'].items():
        spec=suite.tasks[int(task)];env=gym.make(spec.env_id);identities={};shapes=set()
        try:
            for split in ['train','heldout']:
                identities[split]=set()
                for record in data[split]:
                    env.reset(seed=record['seed']);raw=env.unwrapped
                    assert raw.max_steps==data['horizon']==spec.max_steps
                    assert fingerprint(spec.env_id,describe(raw,spec.family))==record['id']
                    identities[split].add(control_identity(raw,spec.family));shapes.add((raw.width,raw.height))
                assert len(identities[split])==len(data[split]), (task,split,'duplicate control states')
            overlap=len(identities['train']&identities['heldout']);assert overlap==0
            assert all(covered(r,training_envelope(data['train'])) for r in data['heldout'])
        finally:env.close()
        rows.append(dict(task=int(task),family=spec.family,env_id=spec.env_id,horizon=spec.max_steps,
            actions=spec.num_actions,canvas=';'.join('%dx%d'%s for s in sorted(shapes)),
            train=len(identities['train']),heldout=len(identities['heldout']),overlap=overlap,
            evaluation='heldout' if data['heldout'] else 'seen_only',
            workers=32,worker_counts=';'.join(str(sum(i%len(data['train'])==j for i in range(32))) for j in range(len(data['train'])))))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=audit(a.bank);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'configuration_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (a.output/'configuration_audit.json').write_text(json.dumps(dict(bank=a.bank.name,bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(),tasks=rows),indent=2)+'\n')
    print('Verified',len(rows),'tasks:',sum(r['train'] for r in rows),'training and',sum(r['heldout'] for r in rows),'heldout configurations; zero overlap.')
