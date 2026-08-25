# Every target writes JSON into runs/ and nothing else, so a crashed sweep
# never costs a figure. Windows: run under Git Bash with make installed, or
# just execute the python lines directly — they are plain module invocations.

PY ?= python
MODELS := gated_1b_baseline gated_1b_headwise gated_1b_elementwise qwen3_0.6b_base gpt2_small
SEEDS  := 0 1 2 3 4
ARMS   := baseline softmax1 gated

.PHONY: test gate measure quant train figs repro clean

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

figs:
	$(PY) -m analysis.aggregate
	$(PY) -m analysis.figures

## Full reproduction. See README for expected runtime and hardware.
repro: gate measure quant train figs

clean:
	rm -rf runs/sinks/* runs/quant/* runs/train/*
