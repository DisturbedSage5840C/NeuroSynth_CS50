from __future__ import annotations

from backend.core.config import get_settings


def _training_pipeline(project_name: str = "neurosynth", model_name: str = "default") -> None:
    from kfp import dsl

    @dsl.component(base_image="python:3.11")
    def train_component(project_name: str, model_name: str) -> None:
        print(f"Running training for {project_name}/{model_name}")

    train_component(project_name=project_name, model_name=model_name)


def trigger_training_run(project_name: str, model_name: str) -> str:
    from kfp import Client

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
    settings = get_settings()
    client = Client(host=settings.kubeflow_host)
    run = client.create_run_from_pipeline_func(
        _training_pipeline,
        arguments={"project_name": project_name, "model_name": model_name},
    )
    return run.run_id
