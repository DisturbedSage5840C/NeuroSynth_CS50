# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Validate that all Part 7 key files (new + modified) from PLAN_V5.md exist.

Usage:
    python scripts/validate_file_manifest.py

Exits 0 when every file is present, 1 if any are missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── New files listed in PLAN_V5.md §7 ────────────────────────────────────────

NEW_FILES = [
    # Data pipeline scripts
    "scripts/data/v5/download_kaggle.py",
    "scripts/data/v5/download_physionet.py",
    "scripts/data/v5/download_uci.py",
    "scripts/data/v5/merge_v5.py",
    "scripts/data/v5/ctgan_augment.py",
    "scripts/data/v5/build_pubmed_corpus.py",
    "scripts/data/v5/embed_corpus.py",
    "scripts/data/v5/scrape_openneuro.py",
    "scripts/data/v5/process_oasis_v5.py",
    "scripts/tune_fusion_weights.py",
    # ML
    "src/neurosynth/models/fusion.py",
    "scripts/train_v5.py",
    "scripts/train_tft_v5.py",
    # Backend
    "backend/report_generator_v4.py",
    "backend/routers/literature.py",
    "backend/routers/data.py",
    "backend/routers/predictions_v3.py",
    "backend/services/data_pipeline_service.py",
    "backend/services/__init__.py",
    "backend/models_v3.py",
    # Frontend — new pages
    "frontend/src/figma-system/app/components/CohortDashboard.tsx",
    "frontend/src/figma-system/app/components/DataPipeline.tsx",
    "frontend/src/figma-system/app/components/LiteratureSearch.tsx",
    "frontend/src/figma-system/app/components/BrainAtlas.tsx",
    "frontend/src/figma-system/app/components/Settings.tsx",
    # Frontend — v3 design system
    "frontend/src/figma-system/app/components/v3/GlassCard.tsx",
    "frontend/src/figma-system/app/components/v3/DataBadge.tsx",
    "frontend/src/figma-system/app/components/v3/RiskChip.tsx",
    "frontend/src/figma-system/app/components/v3/RiskScoreGaugeV3.tsx",
    "frontend/src/figma-system/app/components/v3/SHAPWaterfallV3.tsx",
    "frontend/src/figma-system/app/components/v3/TrajectoryChartV3.tsx",
    "frontend/src/figma-system/app/components/v3/CohortStats.tsx",
    "frontend/src/figma-system/app/components/v3/LiteraturePanel.tsx",
    "frontend/src/figma-system/app/components/v3/ClinicalInput.tsx",
    "frontend/src/figma-system/app/components/v3/SectionHeading.tsx",
    "frontend/src/figma-system/app/components/v3/PulseIndicator.tsx",
    "frontend/src/figma-system/app/components/v3/CytoscapeGraph.tsx",
    "frontend/src/figma-system/app/components/v3/TimelineItem.tsx",
    "frontend/src/figma-system/app/components/v3/v3.css",
    # Frontend — atlas
    "frontend/src/lib/aalAtlas.ts",
    # LandingPage / LoginPage redesign
    "frontend/src/features/auth/landing.css",
    "frontend/src/features/auth/login.css",
    # Deployment
    ".github/workflows/train-validate-v5.yml",
    "frontend/vercel.json",
    # QA
    "tests/integration/test_v3_endpoints.py",
    "scripts/validate_success_metrics.py",
]

# ── Modified files listed in PLAN_V5.md §7 ───────────────────────────────────

MODIFIED_FILES = [
    "backend/db_schema.sql",
    "backend/api.py",
    "backend/requirements.txt",
    "backend/requirements-deploy.txt",
    "src/neurosynth/models/calibrated_ensemble.py",
    "src/neurosynth/temporal_tft/model.py",
    "src/neurosynth/genomic/losses.py",
    "src/neurosynth/genomic/model.py",
    "dvc.yaml",
    "render.yaml",
    "pyproject.toml",
    "DEPLOYMENT.md",
    "CHANGELOG.md",
    "README.md",
    # Frontend
    "frontend/src/figma-system/styles/theme.css",
    "frontend/src/features/auth/LandingPage.tsx",
    "frontend/src/features/auth/LoginPage.tsx",
    "frontend/src/figma-system/app/components/Layout.tsx",
    "frontend/src/figma-system/app/routes.tsx",
    # Celery
    "backend/celery_app.py",
    "backend/tasks.py",
    "backend/routers/__init__.py",
    "backend/routers/health.py",
    "backend/models.py",
]


def check(paths: list[str], label: str) -> list[str]:
    missing = []
    for p in paths:
        full = ROOT / p
        if full.exists():
            print(f"  ✓  {p}")
        else:
            print(f"  ✗  {p}  ← MISSING")
            missing.append(p)
    return missing


def main() -> int:
    print(f"\n{'='*60}")
    print("NeuroSynth v5 — File Manifest Validation")
    print(f"Root: {ROOT}")
    print(f"{'='*60}\n")

    print("NEW FILES:")
    missing_new = check(NEW_FILES, "new")

    print("\nMODIFIED FILES:")
    missing_mod = check(MODIFIED_FILES, "modified")

    total_missing = missing_new + missing_mod
    total = len(NEW_FILES) + len(MODIFIED_FILES)
    present = total - len(total_missing)

    print(f"\n{'='*60}")
    print(f"Result: {present}/{total} files present")
    if total_missing:
        print(f"\nMissing ({len(total_missing)}):")
        for f in total_missing:
            print(f"  - {f}")
        return 1

    print("All files accounted for. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
