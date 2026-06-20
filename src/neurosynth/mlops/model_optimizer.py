from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TritonModelSpec:
    model_repo: Path
    model_name: str
    backend: str
    config_path: Path
    model_path: Path


@dataclass
class LatencyReport:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float


class ModelOptimizer:
    def _config_pbtxt(self, model_name: str = "brain_gnn") -> str:
        return f'''name: "{model_name}"
backend: "pytorch_libtorch"
max_batch_size: 32
input: [
  {{name: "node_features", data_type: TYPE_FP32, dims: [-1, 128]}},
  {{name: "edge_index", data_type: TYPE_INT64, dims: [2, -1]}},
  {{name: "edge_attr", data_type: TYPE_FP32, dims: [-1]}},
  {{name: "batch", data_type: TYPE_INT64, dims: [-1]}}
]
output: [
  {{name: "graph_embedding", data_type: TYPE_FP32, dims: [256]}},
  {{name: "stage_logits", data_type: TYPE_FP32, dims: [3]}},
  {{name: "cdrsb_prediction", data_type: TYPE_FP32, dims: [1]}},
  {{name: "uncertainty", data_type: TYPE_FP32, dims: [1]}}
]
instance_group: [{{kind: KIND_GPU, count: 2}}]
dynamic_batching: {{preferred_batch_size: [8, 16, 32], max_queue_delay_microseconds: 5000}}
'''

    def optimize_gnn_for_triton(self, pytorch_model_path: Path, output_dir: Path) -> TritonModelSpec:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = torch.load(str(pytorch_model_path), map_location="cpu")
        if isinstance(model, torch.nn.Module):
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            n = 116
            e = 400
            example_inputs = (
                torch.randn(n, 128, device=model.device if hasattr(model, "device") else "cuda" if torch.cuda.is_available() else "cpu"),
                torch.randint(0, n, (2, e), dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"),
                torch.randn(e, device="cuda" if torch.cuda.is_available() else "cpu"),
                torch.zeros(n, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"),
            )

            traced = torch.jit.trace(model, example_inputs, strict=False, check_tolerance=1e-4)
            model_repo = output_dir / "models" / "brain_gnn"
            version_dir = model_repo / "1"
            version_dir.mkdir(parents=True, exist_ok=True)
            ts_path = version_dir / "model.pt"
            torch.jit.save(traced, str(ts_path))

            # Optional TensorRT optimization via torch2trt when installed.
            try:
                from torch2trt import torch2trt

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
                trt_model = torch2trt(
                    model,
                    [example_inputs],
                    fp16_mode=True,
                    max_batch_size=32,
                    max_workspace_size=1 << 30,
                )
                plan_path = version_dir / "model.plan"
                torch.save(trt_model.state_dict(), str(plan_path))
            except Exception:
                pass

            cfg = self._config_pbtxt("brain_gnn")
            cfg_path = model_repo / "config.pbtxt"
            cfg_path.write_text(cfg, encoding="utf-8")
            return TritonModelSpec(model_repo=model_repo, model_name="brain_gnn", backend="pytorch_libtorch", config_path=cfg_path, model_path=ts_path)

        raise TypeError("Expected a serialized torch.nn.Module object")

    def benchmark_latency(self, triton_client, model_name: str, n_requests: int = 1000) -> LatencyReport:
        latencies = []
        t0 = time.time()
        for _ in range(n_requests):
            s = time.perf_counter()
            _ = triton_client.infer(model_name=model_name)
            latencies.append((time.perf_counter() - s) * 1000.0)
        elapsed = time.time() - t0

        arr = np.array(latencies, dtype=float)
        return LatencyReport(
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            throughput_rps=float(n_requests / max(elapsed, 1e-6)),
        )
