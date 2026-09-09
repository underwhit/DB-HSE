# Software Manifest

## Core method and baselines

- `scripts/edit_qwen.py`
- `scripts/hs_method.py`
- `scripts/run_dbhse_adaptive.py`
- `scripts/run_dbhse_direct.py`
- `scripts/run_actadd.py`
- `scripts/run_isp.py`
- `scripts/run_prompt_methods.py`

## Data preparation and calibration

- `scripts/prepare_safety_dataset.py`
- `scripts/prepare_mmlu_subset.py`
- `scripts/make_xstest_calibration_split.py`
- `scripts/calibrate_dbhse_gate_on_xstest.py`
- `scripts/prepare_qwen3_14b_splits.py`

## Evaluation and statistical analysis

- `scripts/judge_refusal_cli.py`
- `scripts/compute_paired_refusal_ci.py`
- `scripts/summarize_refusal.py`
- `scripts/summarize_judged_results.py`
- `scripts/analyze_xstest_judged.py`
- `scripts/score_mmlu_outputs.py`
- `scripts/summarize_qwen3_14b_calibration.py`

## Human-audit utilities

- `scripts/make_rr_human_validation_sample.py`
- `scripts/export_xstest_safe_false_refusals.py`
- `scripts/make_xstest_simple_manual_review_jsonl.py`
- `scripts/make_xstest_unsafe_nonrefusal_manual_check_jsonl.py`
- `scripts/make_advbench_nonrefusal_manual_check_jsonl.py`
- `scripts/summarize_xstest_manual_audit.py`
- `scripts/summarize_xstest_heldout_manual_table.py`

## Configuration and examples

- `configs/selected_configurations.json`
- `examples/prompts.jsonl`
- `requirements.txt`
- `README.md`
