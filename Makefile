.PHONY: install test demo demo-build demo-serve pretrain sft rft speedrun

install:
	pip install -e ".[all]"

test:
	pytest tests/ -q

# Visual demo (training curve + base/SFT chat + safety router) — best for employers
demo-build:
	python demo/build_demo.py

demo-serve:
	python demo/serve_demo.py

demo: demo-build
	@echo "Open http://127.0.0.1:8000 after serve starts"
	python demo/serve_demo.py

# Fast scripted demo (train → dashboard → API)
demo-fast:
	bash scripts/demo.sh configs/test.yaml 80 8765

pretrain:
	python -m mythos.train --config configs/shakespeare.yaml --steps 500

pretrain-hf:
	python -m mythos.train --config configs/medium-smoke.yaml --steps 100

protect-main:
	bash scripts/enable-branch-protection.sh

sft:
	python -m mythos.posttrain --config configs/shakespeare.yaml \
		--checkpoint checkpoints/mythos-shakespeare/latest.pt --stage sft --steps 150

rft:
	python -m mythos.posttrain --config configs/shakespeare.yaml \
		--checkpoint checkpoints/mythos-shakespeare/latest.pt --stage rft

speedrun:
	bash scripts/speedrun.sh configs/test.yaml 60
