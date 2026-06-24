.PHONY: install test demo demo-build demo-serve demo-quick pretrain sft rft speedrun

PY := $(shell test -f .venv/bin/python && echo .venv/bin/python || echo python3)

install:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest tests/ -q

# Visual demo — venv-aware; rebuild with stronger demo model (configs/demo.yaml)
demo:
	bash scripts/run-demo.sh

demo-rebuild:
	bash scripts/run-demo.sh --rebuild

demo-build:
	$(PY) demo/build_demo.py

demo-build-quick:
	$(PY) demo/build_demo.py --quick

demo-serve:
	$(PY) demo/serve_demo.py

demo-quick:
	bash scripts/run-demo.sh --quick

# Fast scripted demo (train → dashboard → API)
demo-fast:
	bash scripts/demo.sh configs/test.yaml 80 8765

pretrain:
	$(PY) -m mythos.train --config configs/shakespeare.yaml --steps 500

pretrain-hf:
	python -m mythos.train --config configs/medium-smoke.yaml --steps 100

protect-main:
	bash scripts/enable-branch-protection.sh

sft:
	$(PY) -m mythos.posttrain --config configs/shakespeare.yaml \
		--checkpoint checkpoints/mythos-shakespeare/latest.pt --stage sft --steps 150

rft:
	$(PY) -m mythos.posttrain --config configs/shakespeare.yaml \
		--checkpoint checkpoints/mythos-shakespeare/latest.pt --stage rft

speedrun:
	bash scripts/speedrun.sh configs/test.yaml 60
