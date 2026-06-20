from __future__ import annotations

from dask import delayed

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

def shard_population_matrix(X_population, n_shards: int = 8):
    n = len(X_population)
    step = max(1, n // n_shards)
    return [X_population[i : i + step] for i in range(0, n, step)]


def distributed_batches(X_population, n_shards: int = 8):
    shards = shard_population_matrix(X_population, n_shards=n_shards)
    return [delayed(lambda x: x)(s) for s in shards]
