"""AlexPy 3.1: audit frozen QA 3.0 CI artifacts without retraining or tuning.

The input is a list of GitHub Actions artifact records (source, identity and
raw qa30-results.json). This auditor is deliberately a read-only judge:
it does not change the optimizer's route policy or assume CI timestamps
are request latency. The original CI jobs must be obtained independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

BASELINE = 'c_tree'
ROUTES = ('c_tree', 'bitset', 'python', 'native_set')
METRICS = ('median_ns', 'p95_ns', 'p99_ns', 'peak_bytes')
FEATURES = (('small', 'single'), ('small', 'small'), ('medium', 'wide'))
PLATFORMS = ('ubuntu-latest', 'macos-latest', 'windows-latest')
PYTHONS = ('3.13', '3.15')
ARTIFACT_RE = re.compile(r'^alexpy-qa30-(ubuntu-latest|macos-latest|windows-latest)-py(3\.13|3\.15)-(\d+)$')


def nearest_rank(values, q):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('timing samples are empty')
    return ordered[math.ceil(q * len(ordered)) - 1]


def check_positive(value, where):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{where}: nonpositive/nonfinite value')


def verify_samples(result, where):
    """Independent recomputation of sample quantiles and traced-peak median."""
    samples = result['samples_ns']
    peaks = result['peak_samples']
    if not isinstance(samples, list) or len(samples) < 30:
        raise ValueError(f'{where}: need >=30 samples for tail audit')
    if not isinstance(peaks, list) or len(peaks) < 3:
        raise ValueError(f'{where}: need >=3 isolated peak observations')
    if result['n'] != len(samples):
        raise ValueError(f'{where}: sample count mismatch')
    for v in samples + peaks:
        check_positive(v, where)
    computations = {
        'median_ns': statistics.median(samples),
        'p95_ns': nearest_rank(samples, .95),
        'p99_ns': nearest_rank(samples, .99),
        'peak_bytes': statistics.median(peaks),
    }
    for key, calculated in computations.items():
        check_positive(result[key], where + '/' + key)
        if not math.isclose(result[key], calculated, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f'{where}: inconsistent {key}: {result[key]} != {calculated}')


def feature(n, batch):
    if type(n) is not int or type(batch) is not int or n < 1 or batch < 1:
        raise ValueError('invalid workload dimensions')
    return ('small' if n <= 32 else 'medium' if n <= 1024 else 'large',
            'single' if batch == 1 else 'small' if batch <= 16 else 'wide')


def independent_train_decision(row, cost):
    """Reconstruct QA 3.0 frozen rule solely from independent training processes."""
    sessions = row['sessions']
    if len(sessions) < 3:
        return BASELINE, 'insufficient_processes'
    candidates = []
    for name in ROUTES[1:]:
        if all((values[name]['median_ns'] + cost) <= .95 * values[BASELINE]['median_ns']
               and (values[name]['p99_ns'] + cost) <= values[BASELINE]['p99_ns']
               and values[name]['peak_bytes'] <= values[BASELINE]['peak_bytes']
               for values in sessions):
            candidates.append((statistics.median((values[name]['median_ns'] + cost)
                                                  / values[BASELINE]['median_ns']
                                                  for values in sessions), name))
    return (min(candidates)[1], 'conservative_measured_gain') if candidates else (BASELINE, 'risk_or_gain_gate')


def audit(records, *, required_runs=2):
    """Reject missing/repeated/source-inconsistent evidence, return compact audit.

    Each record: {artifact_id, name, run_id, sha256, source_sha, report}.
    A caller should source artifacts directly from GitHub, not infer reports.
    """
    if not isinstance(records, list) or not records:
        raise ValueError('nonempty records required')
    source_shas = set()
    unique = set()
    coverage = defaultdict(set)
    holdouts = []
    detail = []
    matrix = []
    for record in records:
        name = record['name']
        match = ARTIFACT_RE.fullmatch(name)
        if match is None:
            raise ValueError(f'invalid artifact name: {name}')
        os_name, minor, run_name = match.groups()
        if str(record['run_id']) != run_name:
            raise ValueError('run ID does not match artifact label')
        source_sha = record['source_sha']
        if type(source_sha) is not str or not re.fullmatch(r'[0-9a-f]{40}', source_sha):
            raise ValueError('missing exact source commit')
        if type(record['sha256']) is not str or not re.fullmatch(r'[0-9a-f]{64}', record['sha256']):
            raise ValueError('source digest invalid')
        key = (os_name, minor, run_name)
        if key in unique:
            raise ValueError('duplicate host/Python/run evidence')
        unique.add(key)
        source_shas.add(source_sha)
        coverage[(os_name, minor)].add(run_name)
        report = record['report']
        if report.get('schema') != 'alexpy-qa30/1':
            raise ValueError('unexpected report schema')
        parts = report['environment'].split('|')
        if len(parts) != 5 or parts[2] != run_name or not parts[1].startswith(minor + '.'):
            raise ValueError('environment metadata does not match artifact')
        expected_platform = {'ubuntu-latest': 'Linux', 'macos-latest': 'macOS',
                             'windows-latest': 'Windows'}[os_name]
        if not parts[0].startswith(expected_platform):
            raise ValueError('platform mismatch')
        if report.get('default_promotion') is not False or report.get('experimental') is not True:
            raise ValueError('not experimental/fail closed')
        if 'fresh engine only' not in report.get('switch_scope', ''):
            raise ValueError('live state migration unsupported')
        cost = report['overhead_p99_ns']
        check_positive(cost, 'dispatch overhead')
        sessions = report['sessions']
        if len(sessions) < 3 or len({s['pid'] for s in sessions}) != len(sessions):
            raise ValueError('insufficient independent process IDs')
        train = defaultdict(list)
        for session in sessions:
            if len(session['train']) != len(FEATURES):
                raise ValueError('unexpected train profile count')
            train_keys = set()
            for row in session['train']:
                k = feature(row['n'], row['batch'])
                if k in train_keys:
                    raise ValueError('duplicate train feature')
                train_keys.add(k)
                if set(row['results']) != set(ROUTES):
                    raise ValueError('incomplete route controls')
                for route in ROUTES:
                    verify_samples(row['results'][route], f'{name}/train/{k}/{route}')
                train[k].append(row['results'])
            if train_keys != set(FEATURES):
                raise ValueError('incomplete training features')
        decisions = {(tuple(x['key'])): (x['route'], x['reason']) for x in report['decisions']}
        if set(decisions) != set(FEATURES):
            raise ValueError('incomplete decisions')
        for k in FEATURES:
            derived = independent_train_decision({'sessions': train[k]}, cost)
            if decisions[k] != derived:
                raise ValueError(f'route decision not reproducible from training: {k}')
            matrix.append({'os': os_name, 'python': minor, 'run': run_name,
                           'feature': list(k), 'route': derived[0], 'reason': derived[1]})
        if len(report['holdouts']) != 4:
            raise ValueError('unexpected holdout size')
        for h in report['holdouts']:
            k = feature(h['n'], h['batch'])
            expected = decisions.get(k, (BASELINE, 'unknown_workload'))
            if (h['route'], h['reason']) != expected:
                raise ValueError('holdout route mismatches frozen training')
            if set(h['results']) != set(ROUTES) | {'adaptive'}:
                raise ValueError('incomplete holdout controls')
            for route in (*ROUTES, 'adaptive'):
                verify_samples(h['results'][route], f'{name}/holdout/{k}/{route}')
            base = h['results'][BASELINE]
            adaptive = h['results']['adaptive']
            ratios = {metric: adaptive[metric] / base[metric] for metric in METRICS}
            holdouts.append({'os': os_name, 'python': minor, 'run': run_name,
                             'n': h['n'], 'batch': h['batch'], 'seed': h['seed'],
                             'route': h['route'], 'reason': h['reason'],
                             'adaptive_over_tree': ratios})
        detail.append({'artifact_id': record['artifact_id'], 'name': name,
                       'sha256': record['sha256'], 'os': os_name, 'python': minor,
                       'run_id': run_name, 'selected_routes': sorted(set(x[0] for x in decisions.values()))})
    if len(source_shas) != 1:
        raise ValueError('cannot combine artifacts from different source commits')
    run_sets = [v for v in coverage.values()]
    if set(coverage) != {(os_name, py) for os_name in PLATFORMS for py in PYTHONS}:
        raise ValueError('missing platform/Python cell')
    if any(len(v) != required_runs for v in run_sets) or len(set.union(*run_sets)) != required_runs:
        raise ValueError('missing or unmatched CI runs')
    median_ratios = {m: statistics.median(h['adaptive_over_tree'][m] for h in holdouts)
                     for m in METRICS}
    tail_regressions = sum(h['adaptive_over_tree']['p99_ns'] > 1.05 for h in holdouts)
    # A fail-closed matrix that selects the baseline everywhere is not a new optimization.
    promoted = [m for m in matrix if m['route'] != BASELINE]
    return {'schema': 'alexpy-qa31-crosshost/1', 'source_sha': next(iter(source_shas)),
            'run_ids': sorted(set.union(*run_sets)), 'configs': len(coverage),
            'artifact_count': len(records), 'train_cells': len(matrix),
            'holdout_count': len(holdouts), 'alternative_selections': len(promoted),
            'all_fallback_to_c_tree': not promoted,
            'median_adaptive_over_tree': median_ratios,
            'p95_regressions_over_5pct': sum(h['adaptive_over_tree']['p95_ns'] > 1.05 for h in holdouts),
            'p99_regressions_over_5pct': tail_regressions,
            'max_p99_adaptive_over_tree': max(h['adaptive_over_tree']['p99_ns'] for h in holdouts),
            'auto_promotion_justified': False,
            'conclusion': 'No alternative route passed the training gate; promotion not justified'
                          if not promoted else 'Candidate exists; independent fresh-family gate still needed',
            'no_claims': ['real end-user latency', 'production readiness',
                          'cross-host portability of frozen route choices',
                          'statistically significant tail regression'],
            'artifacts': sorted(detail, key=lambda x: (x['run_id'], x['os'], x['python'])),
            'matrix': sorted(matrix, key=lambda x: (x['run'], x['os'], x['python'], x['feature'])),
            'holdouts': sorted(holdouts, key=lambda x: (x['run'], x['os'], x['python'], x['n']))}


def records_from_manifest(manifest_path, root):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    records = []
    root = Path(root)
    for item in manifest['artifacts']:
        file = root / item['zip_file']
        data = file.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item['sha256']:
            raise ValueError('SHA256 mismatch for ' + item['zip_file'])
        with zipfile.ZipFile(file) as z:
            if z.namelist() != ['qa30-results.json']:
                raise ValueError('unexpected ZIP entries')
            raw = z.read('qa30-results.json')
        if hashlib.sha256(raw).hexdigest() != item['json_sha256']:
            raise ValueError('JSON digest mismatch')
        records.append({**item, 'report': json.loads(raw)})
    result = audit(records, required_runs=manifest['required_runs'])
    if result['source_sha'] != manifest['source_sha']:
        raise ValueError('manifest source commit mismatch')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = records_from_manifest(args.manifest, args.raw_dir)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('artifacts', 'matrix', 'holdouts')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
