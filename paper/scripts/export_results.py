#!/usr/bin/env python3
"""Export measured paper tables/plot data; never rerun inference or alter runs.

Reads immutable campaign artifacts, verifies their bound aggregate hashes,
and independently checks completeness and per-example physical byte matching.
PrefixKV numerical tables precede an optional plotting failure in its original
analyzer; their completion/byte checks are repeated here and that status is
preserved in the provenance ledger, rather than inventing a passed run report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
RUNS = Path('/scratch/djy8hg/runs/mosaickv')
SOURCES = {
    'coco': RUNS / 'coco-shared-32-edb6edf-retry1-20261014/analysis/lower-memory-frontier-21636391/report.json',
    'mile192': RUNS / 'milebench-isoquality-e6-fdeca17-20261020/analysis/larger/report.json',
    'mile256': RUNS / 'milebench-lookm-retry-c4d733b-20261019-v2/analysis/recovered-larger/report.json',
    'prefix': RUNS / 'prefixkv/full-campaign-e4de390-20261019-v6/analysis/final',
}
STORAGE = ['initial_active_kv_storage_bytes', 'final_active_kv_storage_bytes',
           'peak_active_kv_storage_bytes']
NAMES = {'full_kv': 'FullKV', 'prompt_attention_topk': 'Prompt Top-K',
         'lookm_reimpl': 'LOOK-M (R)', 'vl_cache_reimpl': 'VL-Cache (R)',
         'prefixkv_reimpl': 'PrefixKV (R)', 'mosaickv_exact': 'MosaicKV exact'}
bound: dict[str, str] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path: Path) -> None:
    bound[str(path)] = sha(path)


def report(key: str) -> dict:
    path = SOURCES[key]
    d = json.loads(path.read_text())
    assert d['status'] == 'passed' and not d['missing_or_failed'], key
    bind(path)
    for name, expected in d['derived_artifact_hashes'].items():
        p = Path(name)
        assert sha(p) == expected, name
        bind(p)
    return d


def text(path: str, value: str) -> None:
    dest = PAPER / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(value + '\n')


def table(path: str, spec: str, header: str, rows: list[str]) -> None:
    text('generated/' + path, '\n'.join([
        '% Generated from hash-verified measured artifacts by scripts/export_results.py.',
        r'\begin{tabular}{' + spec + '}', r'\toprule', header + r'\\\midrule',
        *[row + r'\\' for row in rows], r'\bottomrule', r'\end{tabular}',
    ]))


def dat(name: str, columns: dict) -> None:
    dest = PAPER / 'generated' / (name + '.dat')
    pd.DataFrame(columns).to_csv(dest, sep=' ', index=False, float_format='%.9g')


def check_rows(rows: pd.DataFrame, n: int, arms: int) -> None:
    assert len(rows) == n * arms
    assert rows.error.isna().all() and (rows.status == 'completed').all()
    keys = ['sample_id', 'method', 'cap_mib'] if 'cap_mib' in rows else ['task', 'lane', 'method', 'sample_id']
    assert not rows.duplicated(keys).any()


def main() -> None:
    (PAPER / 'generated').mkdir(exist_ok=True)
    coco, mile, old = report('coco'), report('mile192'), report('mile256')
    c = pd.DataFrame(coco['aggregates'])
    m = pd.read_parquet(SOURCES['mile192'].parent / 'aggregates.parquet')
    o = pd.read_parquet(SOURCES['mile256'].parent / 'aggregates.parquet')
    mr = pd.read_parquet(SOURCES['mile192'].parent / 'derived_rows.parquet')
    check_rows(mr, 192, 16)
    assert not mr.used_fullkv_fallback.any()
    assert (mr.target_shortfall_bytes.fillna(0) == 0).all()
    for cap, group in mr[mr.method != 'full_kv'].groupby('cap_mib'):
        assert (group[STORAGE[0]] == cap * 2**20).all()
        assert (group.groupby('sample_id')[STORAGE].nunique() == 1).all().all()
    # FullKV was repeated in summary tables for plotting convenience, not inference.
    m = m.drop_duplicates(['method', 'cap_mib', 'metric_name'])
    o = o.drop_duplicates(['method', 'cap_mib', 'metric_name'])
    rows = []
    for method in ['full_kv', 'prompt_attention_topk', 'vl_cache_reimpl', 'mosaickv_exact']:
        r = c[(c.method == method) & (c.cap_mib.isna() if method == 'full_kv' else c.cap_mib == 40)].iloc[0]
        timing = '---' if method == 'full_kv' else f'{r.decode_time:.2f} & {r.end_to_end_time:.2f}'
        if method == 'full_kv': timing = '--- & ---'
        rows.append(f'{NAMES[method]} & {r.metric_value:.2f} & {r[STORAGE[0]] / 2**20:.2f} & {r[STORAGE[1]] / 2**20:.3f} & {timing}')
    table('coco40.tex', 'lrrrrr', r'Method & CIDEr $\uparrow$ & Init. & Final & Decode & E2E', rows)

    for key, frame, caps in [('coco', c, [10, 20, 40, 80]), ('mile192', m, [16, 20, 28, 40, 80]), ('mile256', o, [10, 20, 40, 80])]:
        methods = ['prompt_attention_topk', 'vl_cache_reimpl', 'mosaickv_exact'] if key == 'coco' else ['prompt_attention_topk', 'lookm_reimpl', 'mosaickv_exact']
        all_rows = []
        for method in methods:
            sub = frame[frame.method == method]
            data = {'cap': caps}
            for field in ['metric_value', 'end_to_end_time']:
                if key == 'coco':
                    data['score' if field == 'metric_value' else 'latency'] = [float(sub[sub.cap_mib == cap].iloc[0][field]) for cap in caps]
            if key != 'coco':
                for metric, short in [('rouge_l_f1', 'rouge'), ('accuracy', 'accuracy')]:
                    data[short] = [100 * float(sub[(sub.cap_mib == cap) & sub.metric_name.str.endswith('/' + metric)].iloc[0].metric_value) for cap in caps]
                data['latency'] = [float(sub[sub.cap_mib == cap].iloc[0].end_to_end_time) for cap in caps]
            data['final'] = [float(sub[sub.cap_mib == cap].iloc[0][STORAGE[1]]) / 2**20 for cap in caps]
            dat(key + '_' + method, data)
        for cap in caps:
            for method in methods:
                sub = frame[(frame.method == method) & (frame.cap_mib == cap)]
                r = sub.iloc[0]
                if key == 'coco': quality = f'{r.metric_value:.2f}'
                else:
                    q = {x.metric_name.rsplit('/', 1)[-1]: x.metric_value * 100 for x in sub.itertuples()}
                    quality = f"{q['accuracy']:.2f} & {q['rouge_l_f1']:.2f}"
                all_rows.append(f'{cap} & {NAMES[method]} & {quality} & {r[STORAGE[1]]/2**20:.3f} & {r.end_to_end_time:.2f}')
        if key == 'coco':
            table(key + '_all.tex', 'rlrrr', r'Cap & Method & CIDEr & Final MiB & E2E s', all_rows)
        else:
            table(key + '_all.tex', 'rlrrrr', r'Cap & Method & Acc. (\%) & R-L ($\times100$) & Final MiB & E2E s', all_rows)

    # A compact same-memory MileBench table at the registered anchor budget.
    rows = []
    for method in ['prompt_attention_topk', 'lookm_reimpl', 'mosaickv_exact']:
        sub = m[(m.method == method) & (m.cap_mib == 40)]
        q = {x.metric_name.rsplit('/',1)[-1]:x.metric_value*100 for x in sub.itertuples()}
        rows.append(f"{NAMES[method]} & {q['accuracy']:.2f} & {q['rouge_l_f1']:.2f} & {sub.iloc[0].end_to_end_time:.2f}")
    table('mile40.tex', 'lrrr', r'Method & Acc. (\%) $\uparrow$ & R-L $\uparrow$ & E2E s $\downarrow$', rows)
    phases = mr[(mr.cap_mib == 40) & (mr.method != 'full_kv')].groupby('method')[
        ['prefill_time', 'compression_time', 'decode_time', 'end_to_end_time']].mean()
    table('mile_phases.tex', 'lrrr', r'Method & Compress s & Decode s & E2E s', [
        f'{NAMES[method]} & {phases.loc[method].compression_time:.2f} & {phases.loc[method].decode_time:.2f} & {phases.loc[method].end_to_end_time:.2f}'
        for method in ['prompt_attention_topk', 'lookm_reimpl', 'mosaickv_exact']])
    subtask = mr.copy()
    subtask['subtask'] = subtask.sample_id.str.rsplit('-', n=1).str[0]
    subgroup = subtask[(subtask.cap_mib == 40) | (subtask.method == 'full_kv')].groupby(['subtask', 'method']).task_score.mean()
    table('mile_subtasks.tex', 'lrrrr', r'Subtask & FullKV & Top-K & LOOK-M (R) & MosaicKV', [
        name.replace('-', '{-}') + ' & ' + ' & '.join(f'{100*subgroup.loc[(name, method)]:.2f}' for method in ['full_kv', 'prompt_attention_topk', 'lookm_reimpl', 'mosaickv_exact'])
        for name in ['ActionPrediction', 'ActionSequence', 'CLEVR-Change', 'Spot-the-Diff']])

    pdir = SOURCES['prefix']
    bind(pdir.parent.parent / 'plan.json')
    bind(pdir.parent.parent / 'authorization/full-campaign.json')
    for name in ['sample_rows.parquet', 'metrics.parquet', 'paired_comparisons.parquet']:
        bind(pdir / name)
    p = pd.read_parquet(pdir/'metrics.parquet')
    pr = pd.read_parquet(pdir/'sample_rows.parquet')
    check_rows(pr, 1218, 7)
    for task, n in [('llava_description', 1000), ('mmvet_prefixkv', 218)]:
        assert (pr[pr.task==task].groupby(['lane','method']).size() == n).all()
        shared = pr[(pr.task==task) & (pr.lane=='shared_byte_40mib') & (pr.method!='full_kv')]
        assert (shared.groupby('sample_id')[STORAGE].nunique()==1).all().all()
        assert (shared[STORAGE[0]]==40*2**20).all()
    prefix_rows = []
    for lane in ['paper_native', 'shared_byte_40mib']:
        for task, taskname in [('llava_description','LLaVA-Desc.'), ('mmvet_prefixkv','MM-Vet')]:
            subset = p[(p.task==task)&(p.lane==lane)]
            for method in subset.method.drop_duplicates():
                pair = subset[subset.method==method].set_index('metric_name')
                r = pair.iloc[0]
                prefix_rows.append(f"{taskname} & {'Native' if lane=='paper_native' else '40 MiB'} & {NAMES[method]} & {pair.loc['teacher_forced_ppl'].metric_value:.4f} & {100*pair.loc['rouge_l_f1'].metric_value:.2f} & {r[STORAGE[0]]/2**20:.2f} & {r[STORAGE[2]]/2**20:.2f} & {'---' if method=='full_kv' else f'{r.mean_end_to_end_seconds:.2f}'}")
    table('prefix_all.tex','lllrrrrr',r'Task & Lane & Method & PPL $\downarrow$ & R-L $\uparrow$ & Initial MiB & Peak MiB & E2E s',prefix_rows)
    for lane in ['paper_native','shared_byte_40mib']:
        rows=[]
        for task, taskname in [('llava_description','LLaVA-Desc.'),('mmvet_prefixkv','MM-Vet')]:
            for method in ['prefixkv_reimpl','mosaickv_exact']:
                q=p[(p.task==task)&(p.lane==lane)&(p.method==method)].set_index('metric_name')
                r=q.iloc[0]
                rows.append(f"{taskname} & {NAMES[method]} & {q.loc['teacher_forced_ppl'].metric_value:.4f} & {100*q.loc['rouge_l_f1'].metric_value:.2f} & {r[STORAGE[0]]/2**20:.1f} & {r[STORAGE[2]]/2**20:.1f}")
        table('prefix_'+lane+'.tex','llrrrr',r'Task & Method & PPL $\downarrow$ & R-L $\uparrow$ & Init. & Peak',rows)
    # Boundary plot: long decoding makes initial-cache savings misleading.
    s=p[(p.task=='llava_description')&(p.lane=='paper_native')&(p.metric_name=='rouge_l_f1')].set_index('method')
    dat('prefix_boundaries', {'boundary':[1,2,3], **{method:[s.loc[method, field]/2**20 for field in STORAGE] for method in ['prefixkv_reimpl','mosaickv_exact']}})
    # Preserve every formal cross-budget test, including failures/infeasibility.
    rows=[]
    for x in mile['isoquality_tests']:
        if 'reductions' not in x: continue
        metric='Accuracy' if x['hierarchy_key'].endswith('/accuracy') else 'ROUGE-L'
        r={z['boundary']:z['aggregate_reduction'] for z in x['reductions']}
        rows.append(f"{NAMES[x['baseline_method']]} & {x['baseline_cap_mib']:g}/{x['candidate_cap_mib']:g} & {metric} & {100*r['initial']:.1f} & {100*r['final']:.2f} & {100*x['quality_ci_low']:.2f}, {100*x['quality_ci_high']:.2f} & Fail")
    table('isoquality.tex','lllrrrl',r'Baseline & B/M MiB & Metric & Init. reduction \% & Final/peak \% & Adjusted CI ($\times100$) & Gate',rows)
    # Save full comparisons for review without transporting sample answers/media.
    for name, d in [('coco',coco),('mile192',mile),('mile256',old)]:
        pd.DataFrame(d['comparisons']).to_csv(PAPER/'generated'/f'{name}_comparisons.csv',index=False)
    pd.read_parquet(pdir/'paired_comparisons.parquet').to_csv(PAPER/'generated'/'prefix_comparisons.csv',index=False)
    summary={'schema_version':1,'sources':bound,'checks':{
        'milebench_192_complete':True,'prefixkv_1218_complete':True,
        'shared_initial_final_peak_bytes_match':True,
        'prefixkv_original_plotting_failure':'matplotlib missing; numerical exports complete',
        'lower_memory_equal_quality_milebench':mile['lower_memory_equal_quality_demonstrated'],
        'official_lookm_execution':False,'fullkv_latency_competitor':False},
        'notes':['All figures connect measured points only; no interpolation is evidence.',
                 'The 256- and 192-example MileBench cohorts are reported separately.',
                 'Paired hypothesis intervals are copied from the registered reducers.',
                 'No model inference or benchmark rescoring was performed for this export.']}
    text('generated/provenance.json',json.dumps(summary,indent=2,sort_keys=True))
    print(json.dumps({'status':'passed','source_files':len(bound),'milebench_rows':len(mr),'prefixkv_rows':len(pr)},sort_keys=True))


if __name__=='__main__': main()
