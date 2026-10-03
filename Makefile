ARM ?= b
SEED ?= 0
BUDGET ?= 20

.PHONY: data check targets baseline manual test run audit

data:      ## raw CSV -> typed parquet (md5 + row counts asserted)
	uv run python scripts/convert_raw.py
check:     ## schema checks
	uv run python scripts/check_schema.py
targets:   ## canonical target table
	uv run python scripts/build_targets.py
baseline:  ## E000, both backends
	uv run python scripts/run_baseline.py
manual:    ## Phase 5 hand-run hypotheses
	uv run python scripts/run_manual.py
test:
	uv run pytest -q
run:       ## make run ARM=b SEED=0 BUDGET=3
	uv run python scripts/run_researcher.py --arm $(ARM) --seed $(SEED) --budget $(BUDGET) $(if $(OVERWRITE),--overwrite,)
audit:
	uv run python -m src.tools.audit
