"""Evaluate saved SPS-SQL predictions with the official benchmark evaluator."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys


def bird_worker(payload):
    evaluator, sql, gold, database, qid, timeout = payload
    spec = importlib.util.spec_from_file_location('bird_official', evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.execute_model(sql, gold, database, qid, timeout)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=('bird', 'spider'), required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True, help='Official dev.json')
    p.add_argument('--databases', type=Path, required=True)
    p.add_argument('--evaluator', type=Path, required=True, help='Official evaluation.py')
    p.add_argument('--output', type=Path, required=True, help='New evaluation output directory')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--timeout', type=float, default=30)
    args = p.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        p.error('workers and timeout must be positive')
    predictions = json.loads(args.predictions.read_text())
    rows = json.loads(args.gold.read_text())
    ids = [str(row.get('question_id', i)) for i, row in enumerate(rows)]
    if len(set(ids)) != len(ids) or set(predictions) != set(ids):
        raise ValueError('Prediction IDs must match the complete evaluation cohort')
    if any(not isinstance(sql, str) for sql in predictions.values()):
        raise ValueError('Every prediction must be SQL text (or ERROR)')
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(benchmark=args.benchmark, questions=len(rows),
                      predictions_sha256=digest(args.predictions), gold_sha256=digest(args.gold),
                      evaluator_sha256=digest(args.evaluator))
    if args.benchmark == 'bird':
        tasks = []
        for qid, row in zip(ids, rows):
            database = args.databases / row['db_id'] / (row['db_id'] + '.sqlite')
            if not database.is_file():
                raise FileNotFoundError(database)
            tasks.append((str(args.evaluator.resolve()), predictions[qid], row['SQL'],
                          str(database.resolve()), int(qid), args.timeout))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            scores = list(pool.map(bird_worker, tasks))
        correct = sum(row['res'] for row in scores)
        by_id = {str(row['sql_idx']): row['res'] for row in scores}
        difficulty = {}
        for qid, row in zip(ids, rows):
            cell = difficulty.setdefault(row.get('difficulty', 'unspecified'), {'questions': 0, 'correct': 0})
            cell['questions'] += 1
            cell['correct'] += by_id[qid]
        for cell in difficulty.values():
            cell['ex'] = 100 * cell['correct'] / cell['questions']
        report = dict(**provenance, correct=correct, ex=100 * correct / len(rows),
                      difficulty=difficulty, scores=scores)
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'scores'}, indent=2))
    else:
        # One original database per question: Spider EX. Neither oracle value
        # substitution nor an additional keep_distinct flag is enabled.
        gold = args.output / 'gold.sql'
        pred = args.output / 'predictions.sql'
        gold.write_text(''.join(' '.join(row['query'].split()) + '\t' + row['db_id'] + '\n'
                                for row in rows))
        pred.write_text(''.join(' '.join(predictions[qid].split()) + '\n' for qid in ids))
        command = [sys.executable, str(args.evaluator.resolve()), '--gold', str(gold.resolve()),
                   '--pred', str(pred.resolve()), '--db', str(args.databases.resolve()), '--etype', 'exec']
        with (args.output / 'results.txt').open('w') as log:
            subprocess.run(command, cwd=args.output.resolve(), check=True, stdout=log, stderr=subprocess.STDOUT)
        (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print((args.output / 'results.txt').read_text())


if __name__ == '__main__':
    main()
