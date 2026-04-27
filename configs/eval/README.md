# Eval configs

YAML files in this directory define complete pipeline + eval-set configurations. Each file is loaded by `src.eval.config.load_config(path)` into an `EvalConfig` Pydantic model and consumed by `EvalRunner.run()`.

To author a new config, copy `baseline.yaml`, change the fields you want to vary, and rename. Run with `python -m src.eval.cli run --config configs/eval/<your-config>.yaml`.

See `docs/superpowers/specs/2026-04-26-rag-eval-harness-phase-1-design.md` §7 for the full schema.
