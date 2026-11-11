# SPS-SQL

Code for **SPS-SQL: Shared-Policy Search for Efficient SQL Synthesis**.

## Setup

Use Linux with Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install '.[post,eval]'
```

Download:

- [BIRD](https://bird-bench.github.io/) or [Spider](https://yale-lily.github.io/spider)
  questions, schemas, databases, and the corresponding official evaluator.
- `Qwen/Qwen2.5-Coder-14B-Instruct` into `models/Qwen2.5-Coder-14B-Instruct/`.
- [SPS-SQL-adapters.tar.gz](https://github.com/Avaivartika/SPS-SQL/releases/download/v1.0.1/SPS-SQL-adapters.tar.gz)
  and extract it here to create `adapters/`.

Database paths must follow `DATABASE_ROOT/DB_ID/DB_ID.sqlite`. Keep BIRD's
`database_description/` directories beside the SQLite files.

## Run

In a separate environment with `vllm==0.32.0`, start the generator:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen2.5-Coder-14B-Instruct \
  --served-model-name Qwen/Qwen2.5-Coder-14B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 32 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.82 \
  --seed 42 --enable-prefix-caching --enforce-eager
```

In the client environment, run inference on a second GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python run.py \
  --questions data/bird/dev.json --tables data/bird/dev_tables.json \
  --databases data/bird/dev_databases --output outputs/bird
```

For Spider, substitute its question, schema, and database paths.
Configure search in `configs/search.json` and auxiliary models in
`configs/post.json`. Model paths in the latter are relative to that file.
Predictions are saved in `outputs/bird/final_predictions.json`.

## Evaluate

Supply the benchmark's official evaluator separately:

```bash
python evaluate.py --benchmark bird \
  --predictions outputs/bird/final_predictions.json --gold data/bird/dev.json \
  --databases data/bird/dev_databases --evaluator external/bird/evaluation.py \
  --output outputs/bird-evaluation
```

For Spider, use `--benchmark spider` with its paths and
[official evaluator](https://github.com/taoyds/test-suite-sql-eval).

## License

Apache-2.0. Alpha-SQL-derived assets retain their MIT notice; see `NOTICE`
and `src/sps/prompts/ALPHA_SQL_LICENSE`.
