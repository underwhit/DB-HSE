# Anonymous DB-HSE Software Supplement

This archive contains the implementation and evaluation utilities for
Dual-Branch Hidden-State Extraction (DB-HSE). It is prepared as anonymous
supplementary software for double-blind review.

The archive intentionally excludes model weights, third-party benchmark data,
generated responses, API credentials, and machine-specific paths. Public
datasets should be downloaded from their original sources and placed outside
this directory or under a local `data/` directory.

## Contents

- `configs/selected_configurations.json`: selected model-specific settings and
  the Qwen3-14B configuration-search space reported in the paper.
- `scripts/edit_qwen.py`, `scripts/hs_method.py`: model wrapper and hidden-state
  editing primitive.
- `scripts/run_dbhse_adaptive.py`: adaptive DB-HSE generation.
- `scripts/run_dbhse_direct.py`: ungated query-specific direction used
  for the Qwen3-14B configuration study.
- `scripts/run_actadd.py`, `scripts/run_isp.py`: activation
  steering baselines.
- `scripts/run_prompt_methods.py`: Base and explicit-prompt conditions.
- `scripts/judge_refusal_cli.py`: resumable GPT-4o refusal labeling.
- `scripts/compute_paired_refusal_ci.py`: paired bootstrap confidence intervals
  and exact McNemar tests.
- `scripts/prepare_*.py`, `scripts/make_*.py`, `scripts/summarize_*.py`: dataset
  conversion, split construction, human-audit preparation, and summaries.

The historical experiment logs and superseded one-off scripts are not included.

## Environment

The experiments use Python 3.10 and Hugging Face Transformers. Install a CUDA
build of PyTorch appropriate for the local driver first, then install the
remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The PyTorch command is an example. Use the CUDA wheel matching the target
machine. The reported experiments were run on an NVIDIA A800-80GB GPU.

## Input format

Generation scripts accept JSON or JSONL. Every row must contain `prompt`:

```json
{"source_index": 0, "prompt": "A user query", "label": "unsafe", "should_refuse": true}
```

Additional fields are preserved in generated outputs. The two harmless rows in
`examples/prompts.jsonl` are provided only as a smoke-test schema example.

## Public data

Obtain the following datasets from their official distributions:

- [`harmful_vs_unharmful`](https://huggingface.co/datasets/renjiepi/harmful_vs_unharmful)
  for the main safety evaluation and harmful prototype.
- [XSTest](https://github.com/paul-rottger/exaggerated-safety) for selective
  refusal evaluation.
- [AdvBench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench)
  `harmful_behaviors.csv` for cross-benchmark harmful-query evaluation.
- [MMLU](https://huggingface.co/datasets/cais/mmlu) validation data for utility
  evaluation.

CSV benchmarks can be converted to the common JSONL format:

```bash
python scripts/prepare_safety_dataset.py \
  --input data/harmful_behaviors.csv \
  --output data/advbench_500.jsonl \
  --dataset-name advbench \
  --prompt-column goal \
  --limit 500
```

Construct the deterministic XSTest calibration/evaluation split used by the
additional evaluation:

```bash
python scripts/make_xstest_calibration_split.py \
  --input data/xstest_prompts.csv \
  --output-dir data/xstest_seed13 \
  --seed 17 \
  --safe-size 50 \
  --unsafe-size 50
```

Prepare a deterministic MMLU JSONL subset from an official parquet, CSV, JSON,
or JSONL export:

```bash
python scripts/prepare_mmlu_subset.py \
  --input data/mmlu_validation.parquet \
  --output data/mmlu_seed13_300.jsonl \
  --size 300 \
  --seed 13
```

## Main generation commands

The following Qwen3-8B example shows all five conditions. Replace paths and
settings using `configs/selected_configurations.json` for another model.

```bash
MODEL_ID=Qwen/Qwen3-8B
EVAL=data/main_eval.jsonl
HARMFUL_CAL=data/harmful_calibration.jsonl
OUT=outputs/qwen3_8b
mkdir -p "$OUT"

python scripts/run_prompt_methods.py \
  --model-path "$MODEL_ID" \
  --input "$EVAL" \
  --output-dir "$OUT" \
  --output-prefix qwen3_8b \
  --settings base well_specified \
  --batch-size 16 \
  --max-new-tokens 512

python scripts/run_actadd.py \
  --model-path "$MODEL_ID" \
  --input "$EVAL" \
  --output "$OUT/qwen3_8b_actadd.jsonl" \
  --layer 13 \
  --coeff 1 \
  --batch-size 16 \
  --max-new-tokens 512

python scripts/run_isp.py \
  --model-path "$MODEL_ID" \
  --input "$EVAL" \
  --output "$OUT/qwen3_8b_isp.jsonl" \
  --calibration-input "$HARMFUL_CAL" \
  --calibration-size 200 \
  --layer 3 \
  --scale 1 \
  --batch-size 16 \
  --max-new-tokens 512

python scripts/run_dbhse_adaptive.py \
  --model-path "$MODEL_ID" \
  --input "$EVAL" \
  --output "$OUT/qwen3_8b_dbhse.jsonl" \
  --proto-input "$HARMFUL_CAL" \
  --proto-size 100 \
  --layer 11 \
  --alpha 4 \
  --tau-low 0.6367968320846558 \
  --tau-high 0.8122265338897705 \
  --batch-size 16 \
  --max-new-tokens 512
```

All generation commands use deterministic decoding (`do_sample=False`). The
same runners can evaluate MMLU JSONL because they preserve `ground_truth`; score
the outputs with:

```bash
python scripts/score_mmlu_outputs.py \
  --input outputs/mmlu \
  --out analysis/mmlu_summary.json
```

## Refusal judgment

Set credentials through the environment. Never place an API key in a script:

```bash
export OPENAI_API_KEY='YOUR_KEY'
export JUDGE_MODEL='gpt-4o'

python scripts/judge_refusal_cli.py \
  --input outputs/qwen3_8b \
  --output-dir judged/qwen3_8b \
  --resume
```

`--resume` validates the existing output prefix, skips complete files, and
continues partially judged files.

## Paired uncertainty analysis

Run one model at a time. Each JSONL must contain the same prompts and a refusal
label field such as `is_refusal`:

```bash
python scripts/compute_paired_refusal_ci.py \
  --model Qwen3-8B \
  --target judged/qwen3_8b/qwen3_8b_dbhse.jsonl \
  --comparator Base=judged/qwen3_8b/qwen3_8b_base.jsonl \
  --comparator Prompt-Well=judged/qwen3_8b/qwen3_8b_well_specified.jsonl \
  --comparator ActAdd=judged/qwen3_8b/qwen3_8b_actadd.jsonl \
  --comparator ISP=judged/qwen3_8b/qwen3_8b_isp.jsonl \
  --n-bootstrap 50000 \
  --seed 13 \
  --output analysis/qwen3_8b_pairwise.json
```

The bootstrap samples prompts with replacement and reports DB-HSE minus each
comparator in percentage points. Exact McNemar tests use the paired binary
decisions on the same prompts. To reproduce the table's sequential random
streams when models are run separately, use the model-specific starting seeds
listed in `configs/selected_configurations.json`.

## Human validation and boundary audits

Create the stratified main-experiment validation sample:

```bash
python scripts/make_rr_human_validation_sample.py \
  --inputs judged/qwen25 judged/qwen3 judged/llama3 \
  --output analysis/rr_human_validation_sample.jsonl \
  --markdown-output analysis/rr_human_validation_sample.md \
  --stratify-model \
  --refusal-n 10 \
  --non-refusal-n 10 \
  --seed 42
```

Create XSTest candidate files for manual review:

```bash
python scripts/make_xstest_simple_manual_review_jsonl.py \
  --input judged/xstest/qwen3 \
  --output analysis/xstest_qwen3_safe_refusal_manual.jsonl

python scripts/make_xstest_unsafe_nonrefusal_manual_check_jsonl.py \
  --input judged/xstest/qwen3 \
  --output analysis/xstest_qwen3_unsafe_nonrefusal_manual.jsonl
```

Create an AdvBench audit file containing all automatically identified
non-refusals:

```bash
python scripts/make_advbench_nonrefusal_manual_check_jsonl.py \
  --input judged/advbench \
  --output analysis/advbench_nonrefusal_manual.jsonl
```

## XSTest-specific gate calibration

The calibration script constructs a prototype from unsafe calibration prompts
and estimates gate bounds from the disjoint 50-safe/50-unsafe split:

```bash
python scripts/calibrate_dbhse_gate_on_xstest.py \
  --model-path Qwen/Qwen3-8B \
  --input data/xstest_seed13/xstest_calibration_100.jsonl \
  --output-dir outputs/xstest_gate_qwen3 \
  --layer 11 \
  --prototype-label unsafe \
  --proto-size 50 \
  --model-tag qwen3 \
  --overwrite
```

Use the reported `tau_low` and `tau_high` with
`run_dbhse_adaptive.py` on the disjoint held-out split.

## Qwen3-14B configuration study

The reported Qwen3-14B DB-HSE row evaluates the query-specific direction with
`g=1`. Reproduce its selected setting with:

```bash
python scripts/run_dbhse_direct.py \
  --model-path Qwen/Qwen3-14B \
  --input data/qwen3_14b_shared100.jsonl \
  --output outputs/qwen3_14b_dbhse_l16_a8.jsonl \
  --layer 16 \
  --alpha 8 \
  --batch-size 1 \
  --max-new-tokens 512
```

Layer and scale sweeps use the same command with the values listed in the
configuration file. `summarize_qwen3_14b_calibration.py` applies the stated
selection rule to judged sweep outputs.

## Reproducibility notes

- Layer arguments are one-based hidden-state identifiers where documented by
  each CLI. The wrapper maps them to decoder blocks as implemented in the
  corresponding method script.
- Model files may be supplied as Hugging Face identifiers or local directories.
- `device_map="auto"` is used for portability; reduce batch size if the model is
  offloaded or GPU memory is limited.
- API-based judgments can depend on the externally hosted judge version. The
  human-audit scripts are included to validate refusal-boundary cases.
- This software archive contains no author identity, private server address,
  account name, credential, or absolute local path.
