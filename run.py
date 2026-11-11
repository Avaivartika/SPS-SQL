"""Run SPS-SQL on BIRD/Spider questions and public database schemas."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA_FIELDS = {"db_id", "table_names", "table_names_original", "column_names",
                 "column_names_original", "column_types", "primary_keys", "foreign_keys"}


def write_input(path, content):
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Prepared input changed: {path}; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content)


def prepare(questions, tables, directory):
    raw = questions.read_text()
    rows = json.loads(raw) if raw.lstrip().startswith('[') else [
        json.loads(line) for line in raw.splitlines() if line.strip()]
    public = [{"question_id": r.get("question_id", i), "db_id": r["db_id"],
               "question": r["question"], "evidence": r.get("evidence", "")}
              for i, r in enumerate(rows)]
    if not public or len({q['question_id'] for q in public}) != len(public):
        raise ValueError("Questions must be nonempty and have unique integer IDs")
    for q in public:
        if type(q['question_id']) is not int or any(not isinstance(q[k], str)
               for k in ('db_id', 'question', 'evidence')):
            raise ValueError("Invalid public question fields")
    schema = [{k: v for k, v in row.items() if k in SCHEMA_FIELDS}
              for row in json.loads(tables.read_text())]
    if len({row['db_id'] for row in schema}) != len(schema):
        raise ValueError("Duplicate database schemas")
    if {q['db_id'] for q in public} - {row['db_id'] for row in schema}:
        raise ValueError("A question has no matching schema")
    inputs, catalog = directory / 'questions.jsonl', directory / 'tables.json'
    write_input(inputs, ''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in public))
    write_input(catalog, json.dumps(schema, ensure_ascii=False, indent=2) + '\n')
    return inputs.resolve(), catalog.resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--databases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/search.json')
    p.add_argument('--post-config', type=Path, default=ROOT / 'configs/post.json')
    p.add_argument('--endpoint')
    p.add_argument('--model', help='Served generator model name')
    p.add_argument('--api-key-env', help='Environment variable holding the service key')
    p.add_argument('--no-post', action='store_true', help='Return execution voting without post-correction')
    p.add_argument('--direct', action='store_true', help='Single Generate/Accept attempt without search')
    p.add_argument('--first-round-cache', type=Path)
    p.add_argument('--without-policy-sharing', action='store_true',
                   help='Reuse a matched first round and keep its uniform policy for round two')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    prepared = output.parent / (output.name + '-inputs')
    inputs, tables = prepare(args.questions, args.tables, prepared)
    config = json.loads(args.config.read_text())
    if args.endpoint:
        config['endpoint'] = args.endpoint
    if args.model:
        config['model'] = args.model
    if args.api_key_env:
        config['api_key_env'] = args.api_key_env
    if args.direct:
        config.update(direct_generation=True, population_size=1,
                      search_rounds=1, max_attempt_multiplier=1)
    if args.without_policy_sharing and args.first_round_cache is None:
        p.error('--without-policy-sharing requires --first-round-cache from a matching full run')
    post_path = None
    if not args.no_post:
        post = json.loads(args.post_config.read_text())
        for key in ('base_model', 'structural_checkpoint', 'grounding_checkpoint',
                    'semantic_checkpoint', 'cache_path'):
            if post.get(key) is not None:
                post[key] = str((args.post_config.resolve().parent / post[key]).resolve())
        post['schema_catalog'] = str(tables)
        post_path = prepared / 'post.json'
        write_input(post_path, json.dumps(post, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps({'inputs': str(inputs), 'tables': str(tables),
                          'post_config': str(post_path) if post_path else None}, indent=2))
        return
    from sps.pipeline import run
    result = run(inputs=inputs, output=output, database_root=args.databases.resolve(),
                 post_config=post_path, first_round_cache=args.first_round_cache,
                 freeze_cem_update=args.without_policy_sharing, resume=args.resume, **config)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
