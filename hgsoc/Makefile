.PHONY: env simulate train eval report clean

PYTHON ?= python
VENV ?= .venv
BIN := $(VENV)/bin

env:
	$(PYTHON) -m venv $(VENV)
	. $(BIN)/activate && pip install --upgrade pip setuptools wheel
	. $(BIN)/activate && pip install -e .
	. $(BIN)/activate && pip install numpy pandas scikit-learn matplotlib scipy pytest streamlit pyarrow joblib

DATA_DIR := data
ARTIFACTS_DIR := artifacts

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

simulate: $(DATA_DIR)
	. $(BIN)/activate && python -m hgsoc.simulate --n 500 --seed 1337 --out $(DATA_DIR)/sim_cohort.parquet

train:
	. $(BIN)/activate && python -m hgsoc.models --data $(DATA_DIR)/sim_cohort.parquet --out $(ARTIFACTS_DIR)

eval:
	. $(BIN)/activate && python -m hgsoc.eval --data $(DATA_DIR)/sim_cohort.parquet --models $(ARTIFACTS_DIR) --out reports

report:
	. $(BIN)/activate && python -m hgsoc.report --metrics reports/metrics.json --out reports/README.md

clean:
	rm -rf $(VENV) data artifacts reports
	mkdir -p reports artifacts
	ouch reports/.gitkeep artifacts/.gitkeep
