"""Reproducible fixed-palette banks; finite support is reported rather than padded."""
import argparse
from pathlib import Path
import numpy as np
from projects.TSIL.benchmarks.configurations import describe, fingerprint, write_json
from projects.TSIL.benchmarks.selection_configurations import build_selection_bank, training_envelope, covered
from projects.TSIL.benchmarks.suites import get_tsil_minigrid_suite


def build_retained_bank(suite='mt15_fixed', seed=42, count=32, tasks=None):
    import gymnasium as gym
    if suite not in {'mt15_fixed','mt15_gentle'} or count != 32:
        raise ValueError('Retained protocols require mt15_fixed/mt15_gentle and count=32; finite supports use documented smaller counts')
    specs = get_tsil_minigrid_suite(suite)
    tasks = list(range(15)) if tasks is None else list(tasks)
    if not tasks or len(set(tasks))!=len(tasks) or any(t not in range(15) for t in tasks):
        raise ValueError('Expected unique task IDs 0..14')
    small = {10:8,11:3} if suite == 'mt15_fixed' else {10:32,11:20}
    large = [t for t in tasks if t not in small]
    bank = (build_selection_bank(count=count,seed=seed,tasks=large,suite=suite) if large else
            dict(format_version=1,suite=suite,seed=seed,requested_per_split=count,tasks={}))
    bank['palette_rule'] = 'fixed_semantic_palette_v1'
    bank['identity_rule'] = 'canonical_grid_pose_and_target; no nuisance recoloring or mission paraphrases'
    for task in tasks:
        if task not in small: continue
        spec=specs.tasks[task];env=gym.make(spec.env_id);records={}
        try:
            for offset in range(5000):
                config_seed=seed+task*100000+offset
                env.reset(seed=config_seed);raw=env.unwrapped
                descriptor=describe(raw,spec.family);identity=fingerprint(spec.env_id,descriptor)
                if identity not in records:
                    records[identity]=dict(id=identity,seed=config_seed,descriptor=descriptor,
                        target=float(spec.max_steps),success_count=0,best_success_steps=None)
            if len(records)!=small[task]:
                raise RuntimeError('Task %d: expected %d finite configurations, found %d' % (task,small[task],len(records)))
        finally: env.close()
        records=list(records.values());np.random.RandomState(seed+task).shuffle(records)
        if suite=='mt15_fixed': train,heldout=records,[]
        else:
            train,heldout=records[:len(records)//2],records[len(records)//2:]
            # No policy-based selection; fail if the selected split introduces unseen strata.
            if not all(covered(r,training_envelope(train)) for r in heldout):
                raise RuntimeError('Gentle split lacks marginal coverage')
        bank['tasks'][str(task)]=dict(family=spec.family,env_id=spec.env_id,horizon=spec.max_steps,
            train=train,heldout=heldout,finite_support=small[task],seeds_audited=5000,
            evaluation_scope='seen_only_finite_support' if not heldout else 'disjoint_heldout')
    bank['tasks']=dict(sorted(bank['tasks'].items(),key=lambda r:int(r[0])))
    return bank


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['mt15_fixed','mt15_gentle'],default='mt15_fixed')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',type=int,nargs='+')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.output.exists():p.error('Existing banks are immutable; choose a new output path')
    b=build_retained_bank(a.suite,a.seed,tasks=a.tasks);write_json(a.output,b,compact=True)
    print({t:{k:len(r[k]) for k in ['train','heldout']} for t,r in b['tasks'].items()})
