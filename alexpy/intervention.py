"""Cross-process A/B intervention experiment; never promotes an optimization.

Compares unchanged current sum_squares against two opt-in candidates on
preregistered graph nodes. Measurements are advisory and machine-specific.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from . import _native
from .candidates import CANDIDATES
from .graph_search import State, make_values, mirror_oracles
from .regression_triage import assess

# Deliberately include known regressions, fast native cases and negative controls.
STATES = (
    State('tuple', 1, 'huge'),
    State('list', 1, 'huge'),
    State('tuple', 8, 'huge'),
    State('list', 16, 'acc_overflow'),
    State('list_subclass', 0, 'small'),
    State('list_subclass', 1, 'mixed'),
    State('list_subclass', 8, 'small'),
    State('list', 0, 'small'),
    State('tuple', 0, 'small'),
    State('list', 1, 'small'),
    State('tuple', 1, 'small'),
    State('list', 8, 'small'),
    State('tuple', 1024, 'small'),
    State('list', 10000, 'mixed'),
)


def verify(values):
    """Correctness witnesses execute outside timed region. Exact-int workloads only."""
    a, b, current = mirror_oracles(values)
    if a != b or current != a:
        raise ValueError('baseline mirror divergence')
    for name, fn in CANDIDATES.items():
        if fn(values) != a:
            raise ValueError(f'{name} diverged from exact-int mirrors')
    return hashlib.sha256(str(a).encode('ascii')).hexdigest()


def measure(fn, values, loops):
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn(values)
    return (time.perf_counter_ns() - start) / loops


def worker(states, repeats, target_seconds, session):
    if _native is None or not hasattr(_native, 'sum_squares_bigints'):
        raise RuntimeError('candidate native extension not built')
    rng = random.Random(81583 + session)
    states = list(states)
    rng.shuffle(states)
    rows = []
    for state in states:
        values = make_values(state)
        digest = verify(values)
        for fn in CANDIDATES.values():
            for _ in range(3):
                fn(values)
        rough_ns = max(measure(fn, values, 1) for fn in CANDIDATES.values())
        loops = max(1, min(5000, int(target_seconds * 1e9 / max(1, rough_ns))))
        timings = {name: [] for name in CANDIDATES}
        # Rotate method order per repetition rather than always favoring one path.
        order = list(CANDIDATES)
        for rep in range(repeats):
            current_order = order[rep % len(order):] + order[:rep % len(order)]
            if (rep + session) % 2:
                current_order.reverse()
            for name in current_order:
                timings[name].append(measure(CANDIDATES[name], values, loops))
        ratios = {name: [a/b for a, b in zip(timings['current'], samples)]
                  for name, samples in timings.items() if name != 'current'}
        rows.append({'state': state.id, 'digest': digest, 'loops': loops,
                     'timing_ns': timings, 'ratios': ratios,
                     'session_speedups': {name: statistics.median(pair)
                                          for name, pair in ratios.items()}})
    return {'session': session, 'rows': rows}


def experiment(states=STATES, sessions=3, repeats=9, target_seconds=.004):
    if sessions < 1 or repeats < 1 or target_seconds <= 0:
        raise ValueError('sessions/repeats/target_seconds must be positive')
    if len(set(states)) != len(states) or any(state.phase != 'steady' for state in states):
        raise ValueError('states must be unique steady graph nodes')
    # First gate: no child timing at all if any candidate fails correctness.
    checked = {state.id: verify(make_values(state)) for state in states}
    reports = []
    for session in range(sessions):
        command = [sys.executable, '-m', 'alexpy.intervention', '--worker',
                   '--states', ','.join(s.id for s in states), '--session', str(session),
                   '--repeat', str(repeats), '--target-seconds', str(target_seconds)]
        completed = subprocess.run(command, text=True, check=True, capture_output=True)
        report = json.loads(completed.stdout)
        if {row['state'] for row in report['rows']} != set(checked):
            raise RuntimeError('worker omitted or introduced states')
        if any(row['digest'] != checked[row['state']] for row in report['rows']):
            raise RuntimeError('worker output differs from correctness witness')
        reports.append(report)
    summary = []
    for state in states:
        per_session = [next(r for r in report['rows'] if r['state'] == state.id)
                       for report in reports]
        variants = {}
        for name in ('bigint_native', 'small_dispatch'):
            samples = [s['session_speedups'][name] for s in per_session]
            variants[name] = {'session_median_speedups': samples,
                              'median_speedup': statistics.median(samples),
                              'verdict': assess(samples)}
        summary.append({'state': state.id, 'digest': checked[state.id],
                        'variants': variants})
    # Nothing can silently become the default runtime when a guardset regresses.
    promotion = {}
    for name in ('bigint_native', 'small_dispatch'):
        verdicts = [entry['variants'][name]['verdict'] for entry in summary]
        if 'REPRODUCED_REGRESSION_CANDIDATE' in verdicts:
            promotion[name] = 'BLOCKED_GUARDSET_REGRESSIONS'
        else:
            promotion[name] = 'NOT_PROMOTED_SECOND_MACHINE_REQUIRED'
    report = {'schema': 'alexpy-intervention/v0.1',
              'scope': 'sum_squares only; alternatives not deployed',
              'python': sys.version, 'platform': platform.platform(),
              'generated_utc': datetime.now(timezone.utc).isoformat(),
              'sessions': sessions, 'repeats': repeats,
              'target_seconds': target_seconds, 'rows': summary, 'raw_sessions': reports,
              'promotion_gate': promotion,
              'policy': 'Requires independent confirmation, correctness and no significant guardset regression before promoting any candidate.'}
    blob = json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    report['sha256'] = hashlib.sha256(blob).hexdigest()
    return report


def parse_states(tokens):
    states = []
    for token in tokens.split(','):
        parts = token.split('|')
        if len(parts) != 4:
            raise ValueError('invalid state id')
        states.append(State(parts[0], int(parts[1]), parts[2], parts[3]))
    return states


def main(argv=None):
    parser = argparse.ArgumentParser(description='A/B intervention against unchanged AlexPy')
    parser.add_argument('--sessions', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=9)
    parser.add_argument('--target-seconds', type=float, default=.004)
    parser.add_argument('--json', type=Path, default=Path('intervention-results.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--states', help=argparse.SUPPRESS)
    parser.add_argument('--session', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.states:
            parser.error('--worker requires --states')
        print(json.dumps(worker(parse_states(args.states), args.repeat, args.target_seconds,
                                args.session)))
        return 0
    result = experiment(sessions=args.sessions, repeats=args.repeat,
                        target_seconds=args.target_seconds)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.json), 'states': len(result['rows']),
                      'candidates': {name: {status: sum(r['variants'][name]['verdict'] == status
                                                       for r in result['rows'])
                                         for status in sorted({r['variants'][name]['verdict']
                                                               for r in result['rows']})}
                                     for name in ('bigint_native', 'small_dispatch')}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
