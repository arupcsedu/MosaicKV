#!/usr/bin/env python3
"""Export the completed storage-variant study; never infer or fabricate results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
DEFAULT_RUN = Path('/scratch/djy8hg/runs/evicastkv/coco-256-all-methods-20261106-v3')
PRODUCER = 'a6f689fcdfae2b9023045aa412c6d90d8ce79012'
METHODS = ('full_kv', 'mosaickv_exact', 'prompt_attention_topk',
           'prefixkv_reimpl', 'vl_cache_reimpl', 'lookm_reimpl')
LABELS = ('FullKV reference', r'\textbf{EviCastKV-S}', 'Prompt Top-K (A)',
          'PrefixKV (R+A)', 'VL-Cache (R+A)', 'LOOK-M (R+A)')
SHORT = ('FullKV', 'EviCastKV', 'TopK', 'PrefixKV', 'VLCache', 'LOOKM')
PHASES = {'canary': (6, 6), 'memory': (96, 1536),
          'timing': (16, 1536), 'repeated': (6, 600)}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    target = PROJECT / 'generated' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')


def endpoints(row):
    storage = row.get('storage', {})
    return (row.get('initial_encoded_bytes', storage.get('initial_active_kv_storage_bytes')),
            row.get('final_encoded_bytes', storage.get('final_active_kv_storage_bytes')),
            row.get('decode_owned_peak_bytes'))


def table(headers, rows, align):
    return ('% Exported from hash-verified measured records; do not hand edit.\n'
            '\\begin{tabular}{' + align + '}\n\\toprule\n'
            + ' & '.join(headers) + r'\\' + '\n\\midrule\n'
            + '\n'.join(' & '.join(map(str, row)) + r'\\' for row in rows)
            + '\n\\bottomrule\n\\end{tabular}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    root = args.run
    summary = read(root / 'summary/summary.json')
    plan = read(root / 'plan.json')
    scope = read(root / 'scope.json')
    ledger = read(root / 'ledger.json')
    seal = read(root / 'ledger_seal.json')
    assert summary['source_sha'] == plan['source_sha'] == PRODUCER
    assert seal['ledger_sha256'] == sha(root / 'ledger.json')
    assert all(value is False for value in scope['original_mechanisms'].values())
    assert summary['same_realized_byte_budget'] is False
    assert summary['audited_timing_excluded_from_latency'] is True
    assert summary['global_freshness_certified'] is False
    assert summary['whole_request_KV_peak_certified'] is False
    inputs = {}
    for path in (root / 'plan.json', root / 'scope.json', root / 'ledger.json',
                 root / 'ledger_seal.json', root / 'options.json', root / 'cpu.json',
                 root / 'cpu_tests.txt', root / 'summary/summary.json'):
        inputs[str(path)] = sha(path)
    for filename, expected in summary['data_hashes'].items():
        assert sha(filename) == expected, filename
    phases = {}
    controls = None
    for phase, (worker_count, row_count) in PHASES.items():
        jobs = read(root / phase / 'jobs.json')
        decision = read(root / phase / 'analysis/decision.json')
        records_path = root / phase / 'analysis/all_attempts.json'
        records = read(records_path)
        assert len(jobs) == worker_count and len(records) == row_count
        assert decision['complete'] is True and not decision['errors']
        assert decision['raw_sha256'] == sha(records_path)
        inputs[str(records_path)] = sha(records_path)
        inputs[str(root / phase / 'analysis/decision.json')] = sha(root / phase / 'analysis/decision.json')
        keys = [(r['method'], r['sample_index'], r['trial']) for r in records]
        assert len(keys) == len(set(keys))
        for r in records:
            assert r['status'] == 'completed' and r['source_sha'] == PRODUCER
            assert r['used_fullkv_fallback'] is False
            assert len(r['generated_token_ids']) == 128
            assert isinstance(r['answer']['answer'], str)
            assert math.isfinite(r['end_to_end_seconds']) and r['end_to_end_seconds'] > 0
            initial, final, peak = endpoints(r)
            assert type(initial) is int and type(final) is int and final >= initial > 0
            if phase in ('memory', 'canary'):
                assert type(peak) is int and peak >= final
            if phase == 'canary':
                assert r['audit_fast_128_logits_bitwise_identical'] is True
        for index, job in enumerate(jobs):
            folder = root / phase / f'workers/{index}'
            result = read(folder / 'result.json')
            manifest = read(folder / 'manifest.json')
            assert result['status'] == 'completed' and not result['failures']
            assert result['job'] == job and result['source_sha'] == PRODUCER
            assert result['manifest_sha256'] == sha(folder / 'manifest.json')
            assert result['raw_sha256'] == sha(folder / 'raw.jsonl')
            current = (manifest['model'], manifest['dataset'], manifest['attention'],
                       manifest['precision'], manifest['backend'], manifest['seed'])
            if controls is None:
                controls = current
            assert current == controls
            inputs[str(folder / 'manifest.json')] = sha(folder / 'manifest.json')
            inputs[str(folder / 'result.json')] = sha(folder / 'result.json')
            inputs[str(folder / 'raw.jsonl')] = sha(folder / 'raw.jsonl')
        phases[phase] = records
    memory = {(r['method'], r['sample_index']): r for r in phases['memory']}
    assert set(memory) == {(m, i) for m in METHODS for i in range(2, 258)}
    for phase in ('timing', 'repeated'):
        for row in phases[phase]:
            original = memory[row['method'], row['sample_index']]
            assert row['generated_token_ids'] == original['generated_token_ids']
            assert row['answer'] == original['answer']
            assert endpoints(row)[:2] == endpoints(original)[:2]
            assert row['memory_instrumentation'] is False
    for index in range(2, 258):
        assert len({memory[m, index]['generated_token_ids'][0] for m in METHODS}) == 1
    first_manifest = read(root / 'timing/workers/0/manifest.json')
    expected = {r['method']: r for r in summary['rows']}
    table_rows, data_rows, caption_rows, phase_rows, phase_data = [], [], [], [], []
    for method, label, short in zip(METHODS, LABELS, SHORT):
        measured = [memory[method, i] for i in range(2, 258)]
        timings = [r['end_to_end_seconds'] for r in phases['timing'] if r['method'] == method]
        computed = dict(initial_MiB=statistics.mean(endpoints(r)[0] for r in measured) / 2**20,
                        final_MiB=statistics.mean(endpoints(r)[1] for r in measured) / 2**20,
                        mean_audited_decode_peak_MiB=statistics.mean(endpoints(r)[2] for r in measured) / 2**20,
                        max_audited_decode_peak_MiB=max(endpoints(r)[2] for r in measured) / 2**20,
                        latency_mean_s=statistics.mean(timings),
                        latency_p50_s=float(np.quantile(timings, .5)),
                        latency_p95_s=float(np.quantile(timings, .95)))
        for key, value in computed.items():
            assert math.isclose(value, expected[method][key], abs_tol=1e-9), (method, key)
        e = expected[method]
        table_rows.append((label, f'{e["initial_MiB"]:.2f}',
            f'{e["mean_audited_decode_peak_MiB"]:.2f}/{e["max_audited_decode_peak_MiB"]:.2f}',
            f'{e["final_MiB"]:.2f}', f'{e["cider"]:.2f}',
            f'{e["latency_p50_s"]:.2f}', f'{e["latency_p95_s"]:.2f}'))
        data_rows.append(f'{short} {e["initial_MiB"]:.12f} {e["mean_audited_decode_peak_MiB"]:.12f} '
                         f'{e["final_MiB"]:.12f} {e["cider"]:.12f} {e["latency_p50_s"]:.12f} {e["latency_p95_s"]:.12f}')
        write(f'point_{short}.dat', 'method initial peak final cider p50 p95\n' + data_rows[-1] + '\n')
        answers = [r['answer'] for r in measured]
        caption_rows.append((label, sum(a['answer'] == '' for a in answers),
                             sum(a['output_limit_reached'] for a in answers),
                             f'{statistics.mean(a["answer_tokens"] for a in answers):.2f}'))
        if method != 'full_kv':
            tr = [r for r in phases['timing'] if r['method'] == method]
            phase_keys = ('preprocessing_wall_seconds', 'prefill_construct_retire_wall_seconds',
                          'decode_wall_seconds')
            pv = [statistics.mean(r[k] for r in tr) for k in phase_keys]
            eligible = sum(r['uniform_attention_core']['eligible'] for r in tr)
            assert eligible in (0, 256)
            phase_rows.append((label, *[f'{v:.3f}' for v in pv],
                               f'{e["latency_mean_s"]:.3f}', f'{eligible}/256'))
            phase_data.append(f'{short} ' + ' '.join(f'{v:.12f}' for v in pv)
                              + f' {e["latency_mean_s"]:.12f} {eligible}')
    write('operating_points.tex', table(
        ['Method', 'Initial', r'\shortstack{Decode-owned peak\\mean/max}',
         'Final', r'CIDEr $\uparrow$', r'p50 $\downarrow$', r'p95 $\downarrow$'],
        table_rows, 'lrrrrrr'))
    write('points.dat', 'method initial peak final cider p50 p95\n' + '\n'.join(data_rows) + '\n')
    write('caption_lengths.tex', table(['Method', 'Empty', 'Limit reached', 'Mean scored tokens'],
                                      caption_rows, 'lrrr'))
    write('phases.tex', table(['Method', 'Prepare', 'Prefill/construct/retire', 'Decode',
                              'Full request', 'Replay eligible'], phase_rows, 'lrrrrr'))
    write('phases.dat', 'method prepare prefill decode request eligible\n' + '\n'.join(phase_data) + '\n')
    qrows = []
    for q in summary['quality']:
        label = LABELS[METHODS.index(q['baseline'])]
        qrows.append((label, f'{q["cider_difference"]:+.2f}',
                      f'[{q["paired_95_ci"][0]:+.3f}, {q["paired_95_ci"][1]:+.3f}]',
                      f'[{q["familywise_95_ci"][0]:+.3f}, {q["familywise_95_ci"][1]:+.3f}]'))
    write('quality_intervals.tex', table(['Reference point', r'$\Delta$CIDEr',
                                         'Paired 95\% interval', 'Familywise 95\% interval'], qrows, 'lrrr'))
    lrows, ldata = [], []
    for l in summary['latency']:
        label, short = LABELS[METHODS.index(l['baseline'])], SHORT[METHODS.index(l['baseline'])]
        lo, hi = l['adjusted_ratio_interval']
        lrows.append((label, f'{l["mean_ratio"]:.4f}', f'[{lo:.4f}, {hi:.4f}]',
                      f'{100 * (1-l["mean_ratio"]):.2f}\%'))
        ldata.append(f'{short} {l["mean_ratio"]:.12f} {l["mean_ratio"]-lo:.12f} {hi-l["mean_ratio"]:.12f}')
    write('paired_latency.tex', table(['Baseline', 'Time ratio', 'Adjusted 95\% interval', 'Lower time'], lrows, 'lrrr'))
    write('latency.dat', 'baseline ratio minus plus\n' + '\n'.join(ldata) + '\n')
    evi = expected['mosaickv_exact']
    reductions = []
    for method, label in zip(METHODS, LABELS):
        if method == 'mosaickv_exact':
            continue
        base = expected[method]
        reductions.append((label, *[f'{100*(1-evi[k]/base[k]):.2f}\%'
            for k in ('initial_MiB', 'mean_audited_decode_peak_MiB', 'final_MiB')]))
    write('memory_reductions.tex', table(['Reference point', 'Initial', 'Mean decode peak', 'Final'], reductions, 'lrrr'))
    macros = {'EviInitial': evi['initial_MiB'], 'EviPeak': evi['mean_audited_decode_peak_MiB'],
              'EviMaxPeak': evi['max_audited_decode_peak_MiB'], 'EviFinal': evi['final_MiB'],
              'EviCider': evi['cider'], 'EviMedian': evi['latency_p50_s'], 'EviTail': evi['latency_p95_s'],
              'FullCider': expected['full_kv']['cider'],
              'FullInitialReduction': 100*(1-evi['initial_MiB']/expected['full_kv']['initial_MiB']),
              'FullFinalReduction': 100*(1-evi['final_MiB']/expected['full_kv']['final_MiB']),
              'FullPeakReduction': 100*(1-evi['mean_audited_decode_peak_MiB']/expected['full_kv']['mean_audited_decode_peak_MiB']),
              'TopInitialReduction': 100*(1-evi['initial_MiB']/expected['prompt_attention_topk']['initial_MiB']),
              'TopFinalReduction': 100*(1-evi['final_MiB']/expected['prompt_attention_topk']['final_MiB']),
              'TopPeakReduction': 100*(1-evi['mean_audited_decode_peak_MiB']/expected['prompt_attention_topk']['mean_audited_decode_peak_MiB'])}
    for item in summary['latency']:
        key = dict(prompt_attention_topk='Top', prefixkv_reimpl='Prefix',
                   vl_cache_reimpl='VL', lookm_reimpl='Look')[item['baseline']]
        macros[key + 'Faster'] = 100*(1-item['mean_ratio'])
    write('macros.tex', '\n'.join(f'\\newcommand{{\\{k}}}{{{v:.2f}}}' for k, v in macros.items()) + '\n')
    env = first_manifest['environment']
    versions = env['packages']
    device = first_manifest['gpu']['devices'][0]
    erows = [('Model revision', r'\nolinkurl{' + plan['config']['model']['revision'] + '}'),
             ('COCO revision', r'\nolinkurl{' + plan['config']['dataset']['revision'] + '}'),
             ('Producer revision', r'\nolinkurl{' + PRODUCER + '}'),
             ('Python', '3.11.15'), ('PyTorch / CUDA runtime', f'{versions["torch"]} / {first_manifest["gpu"]["cuda_runtime"]}'),
             ('Transformers / Accelerate', f'{versions["transformers"]} / {versions["accelerate"]}'),
             ('lmms-eval / pycocoevalcap', f'{versions["lmms-eval"]} / {versions["pycocoevalcap"]}'),
             ('GPU / driver', f'{device["name"]} / {device["driver_version"]}'),
             ('Requests / output policy', 'Batch one / 128 predictions, 127 append forwards'),
             ('Processor / dataset split', 'Same model revision / val'),
             ('Seed / attention', '20261106 / eager'),
             ('Installed, not exercised', f'vLLM {versions["vllm"]}; SGLang {versions["sglang"]}')]
    write('environment.tex', table(['Field', 'Executed setting'], erows, 'lp{0.64\linewidth}'))
    review = REPO / 'mosaickv/evicastkv_experiment_prompts'
    reviews = {str(p): sha(p) for p in sorted(review.glob('*.md'))}
    generated = {str(p.relative_to(PROJECT)): sha(p) for p in sorted((PROJECT / 'generated').glob('*'))
                 if p.name != 'provenance.json' and p.is_file()}
    provenance = dict(schema_version=1, experiment_root=str(root), producer_sha=PRODUCER,
        source_inputs_sha256=inputs, review_instruction_files_sha256=reviews,
        original_review_pdf_available=False, review_response_is_paraphrase_not_quotation=True,
        generated_files_sha256=generated, exporter_sha256=sha(Path(__file__)),
        scope='paper_export_only_no_new_inference_no_algorithm_change',
        dataset=plan['config']['dataset'], model=plan['config']['model'],
        metric_recomputation='precomputed complete-corpus CIDEr and draw-specific corpus bootstrap; storage/timing recomputed from raw rows',
        observed_status_counts={phase: dict(Counter(r['status'] for r in rows)) for phase, rows in phases.items()},
        variant='EviCastKV shared-source precision/storage variant',
        byte_matched=False, original_six_mechanisms_active=False,
        whole_request_exact_KV_peak_certified=False, independent_confirmation=False,
        gpu_worker_count=sum(v[0] for v in PHASES.values()),
        historical_results_pooled=False)
    write('provenance.json', provenance)
    print('Verified 124 GPU workers, 1536 memory + 1536 timing + 600 repeated rows; exported measured tables.')
    print('Caption audit (empty / limit / mean scored tokens):', caption_rows)


if __name__ == '__main__':
    main()
