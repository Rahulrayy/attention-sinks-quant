# Every target writes JSON into runs/ and nothing else, so a crashed sweep
# never costs a figure. Windows: run under Git Bash with make installed, or
# just execute the python lines directly — they are plain module invocations.

PY ?= python
MODELS := gated_1b_baseline gated_1b_headwise gated_1b_elementwise qwen3_0.6b_base gpt2_small
SEEDS  := 0 1 2 3 4
ARMS   := baseline softmax1 gated

.PHONY: test gate measure quant dist corpus lambada train figs repro clean

## The Day 3 hard gate. Nothing downstream is valid until this passes.
gate:
	$(PY) -m pytest tests/test_fakequant.py -v

test:
	$(PY) -m pytest tests/ -v

## Track A: sink + massive-activation measurement, both BOS policies.
measure:
	@for m in $(MODELS); do \
	  for s in $(SEEDS); do \
	    $(PY) -m sinks.measure --model $$m --calib-seed $$s --prepend-bos || exit 1; \
	    $(PY) -m sinks.measure --model $$m --calib-seed $$s || exit 1; \
	  done; \
	done

## Track A: the fp16_exceptions x granularity x bits grid.
quant: gate
	@for m in $(MODELS); do \
	  for b in 8 6 4; do \
	    for g in per_tensor per_token; do \
	      for e in none position_0 detected_sinks outlier_channels; do \
	        for s in $(SEEDS); do \
	          $(PY) -m quant.evaluate --model $$m --bits $$b \
	            --act-granularity $$g --fp16-exception $$e --calib-seed $$s \
	            --sinks-json runs/sinks/$${m}_calib$${s}_nobos.json || exit 1; \
	        done; \
	      done; \
	    done; \
	  done; \
	done

## Track B: 3 arms x 5 seeds = 15 runs, ~8-12 h. One overnight.
train:
	@for a in $(ARMS); do \
	  for s in $(SEEDS); do \
	    $(PY) -m train.train --config configs/train/$$a.yaml --seed $$s --resume || exit 1; \
	  done; \
	done

## Track A: per-layer activation shape (R6). Quantizes nothing, so it needs no
## gate and costs one holdout pass per checkpoint.
dist:
	@for m in $(MODELS); do \
	  $(PY) -m quant.distributions --model $$m --bits 8 || exit 1; \
	done
	$(PY) -m analysis.distributions

## The second-corpus arm (R7, C21, C22). All three widths for the grid;
## diagnose and distributions are 8-bit only, which LIMITATIONS §21 records.
## --text-file is the only difference from the `quant` target, and the
## provenance fields (corpus_sha256, holdout_sha) keep the grids distinguishable.
corpus: gate
	@for m in $(MODELS); do \
	  $(PY) -m quant.evaluate --model $$m --grid --bits-list 8,6,4 \
	    --granularities per_tensor,per_token --exceptions none,position_0 --seeds 0 \
	    --seq-len 256 --calib-tokens 2048 --eval-tokens 8192 \
	    --text-file data/code_python.txt --out runs/quant_code || exit 1; \
	  $(PY) -m quant.diagnose --model $$m --bits 8 --skip-coverage \
	    --text-file data/code_python.txt --out runs/diag_code || exit 1; \
	  $(PY) -m quant.distributions --model $$m --bits 8 \
	    --text-file data/code_python.txt --out runs/dist_code || exit 1; \
	done
	$(PY) -m analysis.corpora
	$(PY) -m analysis.corpora --bitwidth

## The one downstream task. Accuracy, not perplexity -- the only metric here
## that can say whether the damage reaches behaviour. --dynamic adds the
## no-calibration control for the calibrate-on-FineWeb / evaluate-on-LAMBADA
## shift. Needs network on first run; the split is fetched, not committed.
lambada: gate
	@for m in $(MODELS); do \
	  $(PY) -m quant.lambada --model $$m --bits 8 --n-examples 1000 --dynamic \
	    || exit 1; \
	done
	$(PY) -m analysis.lambada

figs:
	$(PY) -m analysis.aggregate
	$(PY) -m analysis.figures

## Full reproduction. See README for expected runtime and hardware.
repro: gate measure quant dist corpus lambada train figs

clean:
	rm -rf runs/sinks/* runs/quant/* runs/train/*
