"""Resumable trainer. Checkpoints every 500 steps.

Thermal degradation does not corrupt the result (final loss is the comparison,
not throughput) but a shutdown at hour 6 of a 10-hour sweep does. Hence
checkpointing and clean resume, plus GPU temp/clock logged alongside loss.

Emits one JSON per run to runs/train/<arm>_seed<k>.json.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default="runs/train")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
