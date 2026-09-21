"""
Unified Engine Router for Parallel Constrained Decoding.
"""

from core.engine_torch import (
    get_torch_engine as get_engine,
    run_parallel_generation_torch as run_parallel_generation,
    run_naive_generation_torch as run_naive_generation,
    stream_naive_generation_torch as stream_naive_generation,
)
run_rlcd_generation = run_parallel_generation

__all__ = [
    "get_engine",
    "run_parallel_generation",
    "run_naive_generation",
    "stream_naive_generation",
    "run_rlcd_generation",
]
