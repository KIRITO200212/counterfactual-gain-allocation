#!/usr/bin/env python3
"""Recompute paired scene-bootstrap paper results from numeric-only artifacts.
The summary/contrast functions are copied from the paper analysis, unchanged.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
B=10000
SEED=20270908
SWAP="pixel_pair_slot_swap"
NULL="canonical_resampling_return"

def summary(a, scenes):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    ss = np.asarray(scenes)
    unique = sorted(set(scenes))
    sc = np.array([a[ss == s].mean(axis=0) for s in unique])
    picks = np.random.default_rng(SEED).integers(0, len(sc), (B, len(sc)))
    boot = sc[picks].mean(axis=1)
    def stats(col, point):
        return {'estimate': float(point), 'ci95': np.percentile(col, [2.5, 97.5]).tolist()}
    return sc.mean(axis=0), boot, stats

def tf_contrast(t, c, subset=None):
    assert set(t) == set(c)
    ids = sorted(t if subset is None else set(t) & set(subset))
    if not ids:
        return {'n_rows': 0, 'n_scenes': 0}
    scenes = [t[k]['scene_id'] for k in ids]
    assert all(t[k]['scene_id'] == c[k]['scene_id'] for k in ids)
    fields = [(br, key) for br in [SWAP, NULL, 'factual']
              for key in ['directional_gap', 's_mapped', 's_orig']]
    a = np.array([[t[k]['branches'][br][key] - c[k]['branches'][br][key]
                   for br, key in fields] for k in ids])
    mean, boot, stats = summary(a, scenes)
    result = {'n_rows': len(ids), 'n_scenes': len(set(scenes)),
              'unit': 'mean log-probability per token over the fixed 7-token candidate span'}
    for i, (br, key) in enumerate(fields):
        result[f'delta_{br}_{key}'] = stats(boot[:, i], mean[i])
    g = mean[0] - mean[3]
    gb = boot[:, 0] - boot[:, 3]
    result['d_G_res'] = stats(gb, g)
    result['D_dom'] = stats(-boot[:, 3] - boot[:, 0], -mean[3] - mean[0])
    ratios = -boot[:, 3] / gb
    result['R_null'] = stats(ratios[np.isfinite(ratios)], -mean[3] / g)
    result['null_minus_source_change'] = stats(boot[:, 3] - boot[:, 6], mean[3] - mean[6])
    result['levels'] = {}
    for label, leg in [('control', c), ('candidate', t)]:
        levels = [[leg[k]['branches'][br][key] for br, key in fields] for k in ids]
        lm, _, _ = summary(levels, scenes)
        result['levels'][label] = {f'{br}_{key}': float(lm[i]) for i, (br, key) in enumerate(fields)}
    result['gap_sign_transitions'] = {}
    for br in [SWAP, NULL, 'factual']:
        x = np.array([c[k]['branches'][br]['directional_gap'] for k in ids])
        y = np.array([t[k]['branches'][br]['directional_gap'] for k in ids])
        target = br == SWAP
        old = x > 0 if target else x < 0
        new = y > 0 if target else y < 0
        result['gap_sign_transitions'][br] = {
            'old_correct_sign': int(old.sum()), 'new_correct_sign': int(new.sum()),
            'gained': int((~old & new).sum()), 'lost': int((old & ~new).sum()),
            'note': 'Two-candidate preference sign, not four-way accuracy.'}
    return result

def category(row, gen):
    ans = str(gen.get('parsed_answer') or '').upper()
    if ans not in 'ABCD' or len(ans) != 1 or gen.get('hit_max_new_tokens', False):
        return 'invalid'
    if ans == row['answer_letter'].upper():
        return 'original'
    if ans == row['mapped_answer_letter'].upper():
        return 'mapped'
    return 'other'

def gen_contrast(t, c, branches, subset=None):
    assert set(t) == set(c)
    ids = sorted(t if subset is None else set(t) & set(subset))
    if not ids:
        return {'n_rows': 0}
    scenes = [t[k]['scene_id'] for k in ids]
    assert all(t[k]['scene_id'] == c[k]['scene_id'] for k in ids)
    result = {'n_rows': len(ids), 'n_scenes': len(set(scenes)), 'unit': 'percentage points'}
    for br in branches:
        by_leg = {}
        counts = {}
        for label, leg in [('candidate', t), ('control', c)]:
            matrix = []
            counts[label] = Counter()
            for k in ids:
                b = leg[k]['branches'][br]
                assert b['accepted']
                rolls = b['rollout']['per_rollout']
                assert len(rolls) == b['rollout']['num_generations']
                cats = Counter(category(leg[k], gen) for gen in rolls)
                counts[label].update(cats)
                matrix.append([cats[cat] / len(rolls) for cat in ['original', 'mapped', 'other', 'invalid']])
            by_leg[label] = np.array(matrix)
        m, bs, stat = summary((by_leg['candidate'] - by_leg['control']) * 100, scenes)
        rates = {}
        for label in by_leg:
            rates[label] = summary(by_leg[label] * 100, scenes)[0].tolist()
        result[br] = {'counts': {k: dict(v) for k, v in counts.items()}, 'rates_percent': rates,
                      'deltas': {cat: stat(bs[:, i], m[i]) for i, cat in enumerate(['original', 'mapped', 'other', 'invalid'])}}
    return result

def read_rows(name):
    return {x["sample_id"]: x for x in map(json.loads, (ROOT/"data/numerical"/name).read_text().splitlines())}
def check(actual, expected, path=""):
    if isinstance(expected,dict):
        for k,v in expected.items():
            check(actual[k],v,path+"/"+k)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,v in enumerate(expected):check(actual[i],v,path+f"/{i}")
    elif isinstance(expected,(int,float)):
        assert np.isclose(actual,expected,rtol=0,atol=1e-12),(path,actual,expected)
    else:
        assert actual==expected,(path,actual,expected)
def main():
    subsets=json.loads((ROOT/"data/numerical/subsets.json").read_text())
    expected=json.loads((ROOT/"results/expected_statistics.json").read_text())
    result={"case_A":{},"case_B":{"contrasts":{}}}
    for seed in ["1234","20260910"]:
        reg,con,loc=[read_rows(f"tf_{seed}_{arm}.jsonl") for arm in ["reg","continued","local"]]
        result["case_A"][seed]={
            "primary_contrast":tf_contrast(reg,con),
            "vs_compute_control":tf_contrast(reg,loc),
            "same107_TF_posthoc":tf_contrast(reg,con,subsets["aligned107"]),
            "human_valid_TF_posthoc":tf_contrast(reg,con,subsets["supported"]),
            "primary_generation":gen_contrast(read_rows(f"gen_{seed}_reg.jsonl"),read_rows(f"gen_{seed}_continued.jsonl"),[SWAP,NULL]),
        }
    check(result["case_A"],expected["case_A"],"case_A")
    bridge=json.loads((ROOT/"data/numerical/bridge_scene_observations.json").read_text())
    mats={k:np.array([[r["boundary_probability"],r["suffix_hits"]/r["suffix_total"],r["full_hits"]/r["full_total"]] for r in v]) for k,v in bridge.items()}
    scenes=[r["scene_id"] for r in bridge["full"]]
    for leg in ["continued","prefix_only","theta0"]:
        assert [r["scene_id"] for r in bridge[leg]]==scenes
        m,b,stat=summary(mats["full"]-mats[leg],scenes)
        result["case_B"]["contrasts"]["full_vs_"+leg]={name:stat(b[:,i],m[i]) for i,name in enumerate(["boundary_mapped_probability","suffix_mapped_rate","full_mapped_rate"])}
    check(result["case_B"]["contrasts"],expected["case_B"]["contrasts"],"case_B")
    for leg,rows in bridge.items():
        counts=expected["case_B"]["counts_and_levels"][leg]
        assert sum(r["suffix_hits"] for r in rows)==counts["suffix_mapped"]
        assert sum(r["suffix_total"] for r in rows)==counts["suffix_total"]
        assert sum(r["full_hits"] for r in rows)==counts["full_mapped"]
        np.testing.assert_allclose(mats[leg].mean(axis=0),counts["equal_parent_means"],rtol=0,atol=1e-12)
    (ROOT/"results/reproduced_statistics.json").write_text(json.dumps(result,indent=2)+"\n")
    print("PASS: four TF contrasts and matched generation at both seeds, gross switches, and all local bridge contrasts/counts; tolerance 1e-12.")
if __name__=="__main__":
    main()
