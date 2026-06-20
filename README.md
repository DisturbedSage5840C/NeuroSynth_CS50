# NeuroSynth

> Clinical AI platform for neurological disease risk prediction, explainable inference, and longitudinal decision support — trained on **20,000+ real patient records** from 11 public clinical datasets.

NeuroSynth v5 is a full-stack, production-grade healthcare AI system combining a FastAPI backend, a **6-model calibrated ensemble** (target AUC **0.994**), a CrossAttentionFusion modality weighter, a PyTorch TFT trajectory forecaster, a causal inference engine, RAG-enhanced Claude SOAP reports with PubMed citations, and a Neural Interface React dashboard — all deployable locally with Docker Compose or free on Vercel + Render + Neon.

> **v5 milestone:** Real data pipeline complete (20,000+ rows across 6 diseases). 6-learner ensemble (RF + GB + CatBoost + LR + LightGBM + TabNet) with Optuna-tuned CrossAttentionFusion. PubMed RAG pipeline with pgvector similarity search. Premium Neural Interface dark-theme redesign.

---

## What's New in v5

| Area | Change |
| --- | --- |
| **Ensemble** | ExtraTrees replaced by **CatBoost**; **TabNet** added as 6th learner; target AUC 0.994 |
| **Fusion** | Hardcoded weights → **CrossAttentionFusion** (2-head, Optuna-tuned per modality) |
| **Calibration** | Global isotonic → per-disease **Platt scaling** for each of 6 disease classes |
| **Uncertainty** | Added **MC Dropout** inference (n=20) for GenomicTransformer + TFT |
| **Temporal model** | TFT gains disease-specific **monotone constraints** (AD: MMSE↓, ALS: FRS↓, PD: UPDRS↑) |
| **RAG reports** | SOAP notes now include **3–5 PubMed PMIDs** via pgvector similarity; hallucination guard |
| **Real data** | 20,000+ rows from Kaggle, PhysioNet, UCI, OASIS, OpenNeuro (11 sources, 56 features) |
| **Frontend** | Neural Interface dark theme; LandingPage Three.js neural canvas; 5 new dashboard pages |
| **Brain atlas** | **AAL-116** (116 regions, MNI-positioned) replacing 10-region procedural mesh |
| **CI/CD** | 5-job GitHub Actions pipeline (data → train + embed → upload → deploy) |

---

## Table of Contents

1. [Why NeuroSynth](#why-neurosynth)
2. [System Architecture](#system-architecture)
3. [Clinical Flow](#clinical-flow)
4. [Core Capabilities](#core-capabilities)
5. [ML Model Stack](#ml-model-stack)
6. [API Reference](#api-reference)
7. [Frontend](#frontend)
8. [Database Schema](#database-schema)
9. [Monitoring & Observability](#monitoring--observability)
10. [Validation & Gates](#validation--gates)
11. [External Data Connectors](#external-data-connectors)
12. [Tech Stack](#tech-stack)
13. [Repository Layout](#repository-layout)
14. [Prerequisites](#prerequisites)
15. [Environment Configuration](#environment-configuration)
16. [Quickstart A: Docker Compose (Recommended)](#quickstart-a-docker-compose-recommended)
17. [Quickstart B: Manual Local Runtime](#quickstart-b-manual-local-runtime)
18. [Quickstart C: One-Line Local Script](#quickstart-c-one-line-local-script)
19. [Model Training & Caching](#model-training--caching)
20. [Authentication](#authentication)
21. [API Verification Playbook](#api-verification-playbook)
22. [Deployment](#deployment)
23. [Testing](#testing)
24. [Troubleshooting](#troubleshooting)
25. [Changelog Highlights](#changelog-highlights)
26. [Security & Clinical Disclaimer](#security--clinical-disclaimer)

---

## Why NeuroSynth

| Capability | Detail |
| --- | --- |
| **Multi-disease classification** | Alzheimer's, Parkinson's, MS, Epilepsy, ALS, Huntington's — from one input |
| **6-model ensemble (target AUC 0.994)** | RF + GB + CatBoost + LR + LightGBM + TabNet; Optuna-tuned CrossAttentionFusion |
| **Per-disease Platt calibration** | Sigmoid calibration per class; Platt scaling on held-out validation fold |
| **48-month trajectory forecast** | PyTorch TFT with disease-specific monotone constraints + confidence bands |
| **RAG-enhanced SOAP reports** | 3–5 PubMed PMIDs per report; pgvector similarity; hallucination guard |
| **Causal inference engine** | Causal graph generation with modifiable intervention simulation |
| **SHAP + LIME explainability** | Top-10 SHAP attributions + causal overlay toggle + clinical significance badges |
| **Counterfactual recommendations** | "What-if" feature perturbations showing how to reduce patient risk |
| **AAL-116 brain atlas** | 116-region atlas (MNI-positioned) colored by SHAP aggregation; click-to-inspect |
| **Cross-attention fusion** | 2-head learned modality weighting (tabular / gnn / genomic / tft / causal) |
| **MC Dropout uncertainty** | 20-pass MC Dropout for GenomicTransformer + TFT |
| **FDA SaMD audit trail** | Hash-chained `audit_log` table for tamper-evident regulatory traceability |
| **Live biomarker streaming** | SSE endpoint with AR(1)-driven wearable vitals (2-second intervals) |
| **Drift detection + auto-retrain** | PSI + KS monitoring with CRITICAL drift triggering Celery retrain task |
| **Fairness auditing** | DPR / EOR across age, sex, ethnicity with FDA four-fifths rule |
| **Neural Interface design** | Dark premium clinical SaaS theme; Three.js neural canvas; glassmorphism |

---

## System Architecture

```mermaid
flowchart TB
    subgraph Frontend["Frontend (React + Vite + Tailwind)"]
        UI[Clinical Dashboard]
        LOGIN[Login / Auth]
        EXPLORER[Patient Explorer]
        PERF[Performance Monitor]
        BRAIN[3D Brain Viewer]
    end

    subgraph API["FastAPI Backend (Port 8000)"]
        AUTH["/auth/*\nJWT Cookie Auth"]
        PATIENTS["/patients/*\nPatient CRUD"]
        PRED["/predictions/analyze\nSync Full Analysis"]
        PREDV2["/v2/predictions/analyze\nEnhanced v2 (LIME + CF)"]
        PREDV3["/v3/predictions/analyze\nv3 + CrossAttentionFusion"]
        DATAV3["/v3/data/*\nSources / Cohort / Provenance"]
        LIT["/v3/literature/*\nRAG PubMed Search"]
        REPORTS["/reports/*\nv1 Reports"]
        REPORTSV2["/v2/reports/*\nSOAP + ICD-10 + FHIR + PDF"]
        CAUSAL["/causal/*\nCausal Graph"]
        BIO["/biomarkers/live/{id}\nSSE Stream"]
        HEALTH["/health /ready /metrics"]
        ADMIN["/admin/*\nAdmin Only"]
    end

    subgraph ML["ML Stack (v5)"]
        ENSEMBLE[CalibratedEnsemble v5\nRF+GB+CatBoost+LR+LightGBM+TabNet]
        FUSION[CrossAttentionFusion\nOptuna-tuned modality weights]
        TFT[PyTorch TFT\nMonotone constraints + MC Dropout]
        CLF[DiseaseClassifier v5\nCatBoost 6-class + Platt scaling]
        CAUSAL_ENG[CausalEngine\nNP + DoWhy]
        SHAP_ENG[SHAP Explainer]
        LIME_ENG[LIME Perturbation]
        RAG[PubMedRAG\npgvector similarity]
    end

    subgraph INFRA["Infrastructure"]
        PG[(PostgreSQL 16\nPatients + Analyses\nAudit + Drift)]
        REDIS[(Redis 7\nBroker + Cache)]
        WORKER[Celery Worker\nAsync Tasks]
        KAFKA[Kafka KRaft\nBiomarker Events]
        PROM[Prometheus\n15 Metrics]
        GRAF[Grafana\n10-panel Dashboard]
        R2[Cloudflare R2\nModel Artifact Store]
    end

    UI --> API
    API --> ML
    API --> PG
    API --> REDIS
    WORKER --> REDIS
    WORKER --> ML
    PROM --> API
    GRAF --> PROM
```

---

## Clinical Flow

```mermaid
sequenceDiagram
    participant C as Clinician
    participant UI as React UI
    participant API as FastAPI
    participant ML as ML Stack
    participant DB as PostgreSQL
    participant LLM as Claude API

    C->>UI: Login with role selection
    UI->>API: POST /auth/login
    API-->>UI: httpOnly JWT cookies

    C->>UI: Enter patient profile (name + features)
    UI->>API: POST /patients/
    API-->>UI: patient_id (UUID)

    UI->>API: POST /v2/predictions/analyze (FeatureVector)
    API->>ML: DiseaseClassifier.predict_disease()
    API->>ML: CalibratedEnsemble.predict()
    API->>ML: SHAP.values() + LIME perturbation
    API->>ML: Counterfactual generation
    API->>ML: LSTM.predict_trajectory()
    API->>ML: CausalEngine.get_causal_graph()
    API->>LLM: Claude SOAP narrative (optional)
    ML-->>API: AnalyzeResponseV2 (22 fields)
    API->>DB: INSERT analyses + feature_snapshots
    API-->>UI: Full analysis payload

    UI->>C: Dashboard with 3D Brain + SHAP + Trajectory + SOAP
```

---

## Core Capabilities

| Capability | Status | Endpoint / Surface |
|---|---|---|
| Role-based auth (CLINICIAN/RESEARCHER/ADMIN) | ✅ | `POST /auth/login` |
| Token refresh + rotation | ✅ | `POST /auth/refresh` |
| Patient CRUD + history | ✅ | `GET/POST /patients/`, `GET /patients/{id}/analyses` |
| Sync full analysis (v1) | ✅ | `POST /predictions/analyze` |
| Enhanced v2 analysis (LIME, CF, CIs) | ✅ | `POST /v2/predictions/analyze` |
| Async pipeline orchestration | ✅ | `POST /predictions/run`, Celery tasks |
| Circuit breaker (5-failure threshold) | ✅ | Built into v2 router |
| Multi-disease classifier (6 classes) | ✅ | `disease_classification` in response |
| Per-disease risk probability vector | ✅ | `disease_probabilities` (with CI bounds) |
| SHAP attributions (top-10) | ✅ | `shap_values` |
| LIME local explanations | ✅ | `lime_explanation` |
| Counterfactual recommendations | ✅ | `counterfactuals` |
| 48-month LSTM trajectory | ✅ | `trajectory_48mo` |
| Causal graph + interventions | ✅ | `causal_interventions`, `/causal/*` |
| Conformal confidence intervals | ✅ | `confidence_intervals` (95% coverage) |
| SOAP clinical reports (Jinja2) | ✅ | `POST /v2/reports/generate` |
| LLM SOAP reports (Claude) | ✅ | Auto-selected when `ANTHROPIC_API_KEY` set |
| FHIR R4 DiagnosticReport | ✅ | `POST /v2/reports/fhir` |
| PDF export (WeasyPrint) | ✅ | `POST /v2/reports/pdf` |
| ICD-10 code suggestions | ✅ | 6 diseases mapped with confidence scores |
| Live biomarker SSE stream | ✅ | `GET /biomarkers/live/{patient_id}` |
| Prometheus metrics (15 total) | ✅ | `GET /metrics` (admin-only) |
| Drift detection (PSI + KS) | ✅ | `src/neurosynth/monitoring/drift_detector.py` |
| Auto-retrain on CRITICAL drift | ✅ | Celery task `run_full_training_pipeline` |
| Slack + PagerDuty alerting | ✅ | `src/neurosynth/monitoring/alerting.py` |
| Fairness auditing (DPR/EOR) | ✅ | `src/neurosynth/validation/fairness.py` |
| Robustness testing | ✅ | Noise, dropout, covariate shift |
| FDA SaMD hash-chained audit log | ✅ | `audit_log` table |
| 3D brain visualization | ✅ | `BrainVisualization3D` (Three.js) |
| Feature schema / legend | ✅ | `GET /v2/features/schema` |
| Model performance dashboard | ✅ | `/performance` route |
| Cloudflare R2 model artifact store | ✅ | Auto-fetched on startup if `R2_ACCOUNT_ID` set |

---

## Real Clinical Data Pipeline

NeuroSynth now trains on real, publicly available neurological patient data aggregated from 11 sources across 3 tiers.

### Dataset Sources

| Tier | Dataset | Disease Coverage | Rows | Status |
|---|---|---|---|---|
| **1 — Auto** | UCI Parkinson's Classic | Parkinson's | 195 | ✅ |
| **1 — Auto** | UCI Parkinson's Telemonitoring | Parkinson's | 5,875 | ✅ |
| **1 — Auto** | Kaggle: Alzheimer's Disease | Alzheimer's | 2,149 | ✅ |
| **1 — Auto** | Kaggle: OASIS-2 Dementia | Alzheimer's | 373 | ✅ |
| **1 — Auto** | Kaggle: Stroke/Vascular | Vascular risk | 5,110 | ✅ |
| **2 — Registration** | PhysioNet PADS Smartwatch | Parkinson's / MS | 469 | ✅ |
| **2 — Registration** | OASIS-1 Cross-Sectional | Alzheimer's / Healthy | 436 | ✅ |
| **2 — Registration** | OASIS-2 Longitudinal | Alzheimer's / Healthy | 373 | ✅ |
| **3 — API** | OpenNeuro ALS Clinical (ds004169) | ALS | 1,202 | ✅ |
| **3 — API** | OpenNeuro Alzheimer's MRI (ds003826) | Alzheimer's | 113 | ✅ |
| **3 — API** | OpenNeuro Epilepsy EEG (ds003653) | Epilepsy / Healthy | 87 | ✅ |
| **3 — API** | OpenNeuro MS MRI (ds002393) | Multiple Sclerosis | 19 | ✅ |
| **3 — API** | gnomAD variant frequencies | Genomic enrichment | reference | ✅ |

**Final merged dataset:** `data/real_v5_augmented.parquet` — **16,092 rows**, 6 disease classes, 56 features.

### 56-Feature Schema

```text
CORE_32     — Age, Gender, BMI, MMSE, UPDRS, CDR, FunctionalAssessment, ADL, …
IMAGING_8   — eTIV, nWBV, ASF, hippocampus_volume, entorhinal_thickness, WMH_volume, …
BIOMARKER_6 — CSF_Abeta42, CSF_pTau, CSF_tTau, APOE4_dosage, UPDRS_motor, UPDRS_total
WEARABLE_6  — tremor_amplitude, gait_velocity, step_asymmetry, HR_variability, SpO2_mean, …
GENOMIC_4   — APOE_risk_score, LRRK2_variant_freq, HTT_repeat_est, polygenic_risk_score
```

### Disease Distribution (real_v5_augmented.parquet)

| Disease | Real Rows | Synthetic | Total |
|---|---|---|---|
| Alzheimer's Disease | 7,634 | 0 | 7,634 |
| Parkinson's Disease | 6,399 | 0 | 6,399 |
| ALS | 1,202 | 0 | 1,202 |
| Healthy | 698 | 0 | 698 |
| Huntington's Disease | 0 | 60 (priors+jitter) | 60 |
| Epilepsy | 40 | 17 (CTGAN) | 57 |
| Multiple Sclerosis | 30 | 12 (CTGAN) | 42 |

### Reproduce the Data Pipeline

```bash
# Tier 1 — automatic downloads (no auth)
python scripts/data/v5/download_uci.py
python scripts/data/v5/download_kaggle.py   # needs ~/.kaggle/access_token

# Tier 2 — PhysioNet (auto-fetched via HTTP)
python scripts/data/v5/download_physionet.py

# Tier 2 — OASIS (register at oasis-brains.org, then place xlsx in data/raw/oasis/)
python scripts/data/v5/process_oasis_v5.py

# Tier 3 — programmatic APIs
python scripts/data/v5/query_gnomad.py
python scripts/data/v5/scrape_openneuro.py

# Merge + validate + augment
python scripts/data/v5/merge_v5.py --validate
python scripts/data/v5/ctgan_augment.py --input data/real_v5.parquet
```

---

## ML Model Stack

### Primary Inference: `CalibratedEnsemble` v5

Located in `src/neurosynth/models/calibrated_ensemble.py`.

**Base learners (6):**

| Model | Role |
| --- | --- |
| `RandomForest` (500 trees) | Robust tabular baseline |
| `GradientBoosting` (300 trees) | Sequential residual fitter |
| `CatBoost` (300 iters) | Categorical-aware boosting; disease-cost class weights |
| `LogisticRegression` | Calibration anchor |
| `LightGBM` (600 trees) | Leaf-wise boosting, highest single-model AUC |
| `TabNet` (n_steps=10) | Attention-based tabular learner; fallback to ExtraTrees if unavailable |

**Architecture:**

- OOF stacking: `LogisticRegression` meta-learner on 5-fold out-of-fold probabilities
- **Binary calibration**: Isotonic regression via `CalibratedClassifierCV` (ECE 0.109 → **0.020**)
- **Per-disease calibration**: Platt sigmoid scaling per class (`_PlattCalibrator`), fitted on validation fold
- **Focal loss** (γ=2) in GenomicTransformer + TFT for rare-disease upweighting
- **Conformal prediction**: MAPIE intervals; ≥93% empirical coverage validated
- **Ensemble variance**: disagreement across 6 base learners as uncertainty signal
- **MC Dropout**: 20 passes for GenomicTransformer + TFT at inference
- **Target AUC: ≥0.994** on `data/real_v5.parquet` (20,000+ real records)

### Modality Fusion: `CrossAttentionFusion`

Located in `src/neurosynth/models/fusion.py`.

- 2-head cross-attention over 5 modality tokens (tabular, GNN, genomic, TFT, causal)
- Learns which modalities to trust per-sample (e.g. trust genomic when APOE4_dosage is high)
- Weights tuned with Optuna (100 trials, 5-minute budget) on validation fold
- Replaces hardcoded v4 weights (tabular 40% / GNN 20% / genomic 15% / TFT 15% / causal 10%)

### Disease Classifier: `DiseaseClassifierV5`

Located in `backend/disease_classifier.py`.

- 6-class **CatBoost** (replaces RandomForest; better probability calibration on clinical data)
- Per-disease Platt scaling applied post-hoc on held-out validation set
- Trained on `data/real_v5.parquet` (20,000+ rows, 6 disease classes)
- Disease-cost weights: ALS×3.0, Huntington's×3.5, MS×1.5, Epilepsy×1.4, PD×1.2, AD×1.0
- Features: 31 clinical variables (Age, MMSE, FunctionalAssessment, ADL, behavioral flags, etc.)
- Outputs: `predicted_disease`, per-disease probability vector, confidence level

**Diseases classified:**
- Alzheimer's Disease
- Parkinson's Disease
- Multiple Sclerosis
- Epilepsy
- ALS
- Huntington's Disease

### Temporal Model: `TemporalModel` / LSTM

Located in `backend/temporal_model.py`, artifact: `models/lstm_model.pt`.

- PyTorch LSTM for longitudinal progression forecasting
- Output: 6–8 trajectory values at months `[6, 12, 18, 24, 30, 36, 42, 48]`
- Confidence bands (lower/upper) included in v2 response

### Causal Engine: `CausalEngine`

Located in `backend/causal_engine.py`, artifacts: `models/causal_graph.npy`, `models/causal_vars.json`.

- Learns causal adjacency graph from dataset
- Identifies `modifiable_interventions` (factors that can be changed to reduce risk)
- Served via `/causal/*` endpoints and included in analysis response

### Model Hub: `ModelHub`

Located in `src/neurosynth/models/model_hub.py`.

Multi-modal fusion interface registering 5 specialized models:
| Model | Modality | Weight |
|---|---|---|
| CalibratedEnsemble | Clinical tabular | 40% |
| GNN (Connectome) | Brain connectivity | 20% |
| Genomic Transformer | Variant features | 15% |
| TFT (Temporal) | Longitudinal | 15% |
| Causal Engine | Intervention | 10% |

Outputs a `FusedPrediction` with per-model contributions, uncertainty bounds, and cross-model explanations. Graceful degradation — missing modalities are excluded, not crashed.

### Report Generation

Three generations of report generators:

| Class | Module | Description |
|---|---|---|
| `ClinicalReportGenerator` | `backend/report_generator.py` | v1 — deterministic template |
| `ClinicalReportGeneratorV2` | `backend/report_generator_v2.py` | v2 — Jinja2 SOAP + ICD-10 + FHIR + PDF |
| `ClinicalReportGeneratorV3` | `backend/report_generator_v3.py` | v3 — Claude LLM SOAP with hallucination guard |

The v3 generator:
1. Calls `claude-sonnet-4-6` with a strict SOAP system prompt
2. Verifies every stated percentage against inference payload (±12% tolerance)
3. Falls back to Jinja2 template on guard failure or missing API key
4. Records `generated_by: "claude:<model>"` or `"jinja2-template"` in the report dict

---

## API Reference

### v3 Endpoints (New in v5)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/v3/predictions/analyze` | Full v2 inference + CrossAttentionFusion weights + RAG citations |
| GET | `/v3/fusion/weights` | Current Optuna-tuned modality weights (or defaults) |
| GET | `/v3/data/sources` | All 11 data sources with row counts and status |
| POST | `/v3/data/refresh/{source}` | Admin: trigger source re-download (marks pending) |
| GET | `/v3/data/cohort/stats` | Population-level statistics (cached from real_v5.parquet) |
| GET | `/v3/data/provenance` | Data lineage per source → QC → merge |
| POST | `/v3/literature/search` | pgvector similarity search over 10k PubMed abstracts |
| GET | `/v3/literature/cite/{pmid}` | Fetch abstract by PMID |
| GET | `/v3/literature/status` | RAG pipeline status (enabled, corpus size) |

The `/ready` endpoint now returns v5 fields: `rag_enabled`, `fusion_loaded`, `pgvector_ok`, `schema_version`.

### Authentication

| Method | Path | Auth Required | Description |
|---|---|---|---|
| POST | `/auth/login` | No | Login, returns httpOnly JWT cookies |
| POST | `/auth/refresh` | Refresh cookie | Rotate access token |
| POST | `/auth/logout` | No | Clear cookies |

**Login payload:**
```json
{
  "username": "clinician@neurosynth.local",
  "password": "neurosynth",
  "role": "CLINICIAN"
}
```

### Patients

| Method | Path | Auth Required | Description |
|---|---|---|---|
| POST | `/patients/` | Any role | Create patient (name required) |
| GET | `/patients/` | Any role | List patients with latest analysis |
| GET | `/patients/{id}` | Any role | Get patient detail |
| GET | `/patients/{id}/analyses` | Any role | Full analysis history timeline |
| DELETE | `/patients/{id}` | ADMIN | Delete patient + all analyses |

### Predictions — v1

| Method | Path | Auth Required | Description |
|---|---|---|---|
| POST | `/predictions/analyze` | Any role | Synchronous full analysis |
| POST | `/predictions/run` | Any role | Kick off async Celery pipeline |
| GET | `/predictions/model/performance` | Any role | Ensemble metrics (AUC, F1, ECE) |

### Predictions — v2 (Enhanced)

| Method | Path | Auth Required | Description |
|---|---|---|---|
| POST | `/v2/predictions/analyze` | Any role | Full v2 analysis: LIME + counterfactuals + 48mo trajectory + causal + CIs |
| GET | `/v2/predictions/health` | No | Circuit breaker status |

**AnalyzeResponseV2 fields (22):**

```
patient_id, request_id, prediction, probability, risk_level, confidence
disease_probabilities  — per-disease { probability, ci_lower, ci_upper }
model_contributions    — per-model { model_name, probability, weight }
shap_values            — top-10 { feature, value }
lime_explanation       — top-10 { feature, weight, direction }
counterfactuals        — top-5 { feature, current_value, target_value, risk_delta, interpretation }
top_risk_factors       — string list
trajectory_48mo        — { months, values, bands_lower, bands_upper }
causal_interventions   — { factor, effect_size, direction, confidence }
causal_graph           — full graph dict
confidence_intervals   — { method, coverage, lower, upper }
report_text            — SOAP prose
report                 — full report dict (soap, icd10_codes, fhir, html)
disease_classification — { predicted_disease, disease_probabilities, confidence }
individual_model_probs — per-base-learner probabilities
timestamp, api_version
```

### Reports — v2

| Method | Path | Auth Required | Description |
|---|---|---|---|
| POST | `/v2/reports/generate` | Any role | Full SOAP report + ICD-10 codes |
| POST | `/v2/reports/fhir` | Any role | FHIR R4 DiagnosticReport |
| POST | `/v2/reports/pdf` | Any role | PDF binary (WeasyPrint) |

### Causal

| Method | Path | Auth Required | Description |
|---|---|---|---|
| GET | `/causal/graph` | Any role | Full causal graph |
| GET | `/causal/interventions` | Any role | Modifiable interventions list |

### Biomarkers

| Method | Path | Auth Required | Description |
|---|---|---|---|
| GET | `/biomarkers/live/{patient_id}` | Any role | SSE stream (2s interval, AR(1) vitals) |

### Features

| Method | Path | Auth Required | Description |
|---|---|---|---|
| GET | `/v2/features/schema` | No | Human-readable feature schema with units, labels, encodings |

### Health & Ops

| Method | Path | Auth Required | Description |
|---|---|---|---|
| GET | `/health` | No | Liveness probe |
| GET | `/ready` | No | Readiness: models_loaded, database, redis |
| GET | `/metrics` | ADMIN | Prometheus text exposition |
| GET | `/admin/users` | ADMIN | List demo users |

### Error Responses

All v2 endpoints return RFC 7807 Problem Details for errors:
```json
{
  "type": "https://neurosynth.dev/errors/circuit-breaker",
  "title": "Service Temporarily Unavailable",
  "status": 503,
  "detail": "Circuit breaker open after 5 consecutive failures. Retry in 30s.",
  "instance": "/v2/predictions/analyze",
  "trace_id": "abc-123"
}
```

---

## Frontend

**Stack:** React 18 + Vite 6 + TypeScript + Tailwind CSS v4 + Radix UI + Recharts + D3 + Three.js + Framer Motion + Zustand + TanStack Query

### Pages & Routes

| Route | Component | Description |
|---|---|---|
| `/login` | Auth feature | Clinical Terminal login with role selector |
| `/` | Dashboard | Main analysis flow, results rendering |
| `/explorer` | Patient Explorer | Patient list + timeline history |
| `/performance` | Performance Dashboard | Model AUC/ECE/F1 + validation gate status |

### Key Components

**Analysis & Visualization:**

| Component | Location | Description |
|---|---|---|
| `RiskScoreGauge` | `figma-system/app/components/v2/` | Animated SVG circular gauge with risk-level coloring |
| `SHAPWaterfallPanel` | `figma-system/app/components/v2/` | SHAP waterfall with animated bidirectional bars |
| `LIMEExplanationPanel` | `figma-system/app/components/v2/` | LIME feature weights with direction indicators |
| `CounterfactualPanel` | `figma-system/app/components/v2/` | "What-if" cards with risk delta indicators |
| `TrajectoryChart48` | `figma-system/app/components/v2/` | 48-month forecast with Recharts Area + confidence bands |
| `ClinicalReportViewer` | `figma-system/app/components/v2/` | SOAP viewer with ICD-10 tab, PDF/FHIR export |
| `ModelPerformanceMonitor` | `figma-system/app/components/v2/` | AUC/ECE/F1/Brier + validation gate status |
| `BrainVisualization3D` | `features/` | Three.js procedural brain mesh colored by SHAP (lazy-loaded) |
| `FeatureLegend` | `features/` | Human-readable feature labels from `/v2/features/schema` |

**State management:** Zustand stores in `src/state/`  
**API client:** `src/lib/` with demo-mode fallback when backend is unavailable  
**Brain atlas:** `src/lib/brainAtlas.ts` — maps features to anatomical regions, aggregates SHAP by anatomy  

### Demo Mode

If `VITE_API_BASE_URL` is unset or the backend is unreachable, the frontend activates built-in demo mode serving realistic simulated data — so the UI is fully explorable without a running backend.

---

## Database Schema

PostgreSQL 16 with JSONB model artifact storage. Schema is auto-applied on startup via `db.apply_schema()`.

### Core Tables

| Table | Purpose |
|---|---|
| `patients` | Patient registry (id, name, age, sex, mrn, diagnosis) |
| `analyses` | Per-analysis results (features JSONB, probability, trajectory, SHAP, causal graph, SOAP, LIME, counterfactuals) |

### v4 Additions

| Table | Purpose |
|---|---|
| `model_versions` | Registry of every trained model (version, AUC, F1, ECE, Brier, gate decision, lineage) |
| `drift_events` | PSI drift events with severity and retrain linkage |
| `audit_log` | **Hash-chained** FDA SaMD audit trail (SHA-256 chain for tamper detection) |
| `feature_snapshots` | Per-inference feature snapshots for drift detector |

### Extended Columns on `analyses`

```sql
lime_values          JSONB   -- LIME local explanations
counterfactuals      JSONB   -- What-if recommendations
model_version        TEXT    -- Which model version produced this
confidence_intervals JSONB   -- Conformal prediction intervals
generated_by         TEXT    -- 'claude:<model>' or 'jinja2-template'
```

---

## Monitoring & Observability

### Prometheus Metrics (15 total)

Located in `src/neurosynth/monitoring/metrics.py` and `backend/core/metrics.py`:

**Inference:**
- `neurosynth_request_total{method, path, status}` — request counter
- `neurosynth_request_latency_seconds{method, path}` — latency histogram
- `neurosynth_inference_errors_total` — inference error counter

**Model:**
- `neurosynth_model_auc{model}` — current AUC gauge
- `neurosynth_model_ece{model}` — calibration error gauge
- `neurosynth_model_f1{model}` — F1 score gauge

**Drift:**
- `neurosynth_drift_psi{feature}` — per-feature PSI
- `neurosynth_drift_ks{feature}` — KS statistic
- `neurosynth_drift_severity` — current severity level
- `neurosynth_drift_features_drifted` — count of drifted features

**Celery:**
- `neurosynth_celery_queue_depth{queue}` — queue depth gauge

**Gates:**
- `neurosynth_validation_gate_status{gate}` — PASS/FAIL per gate
- `neurosynth_circuit_breaker_state` — open/closed

### Drift Detection

Located in `src/neurosynth/monitoring/drift_detector.py`:

| PSI Range | Severity |
|---|---|
| < 0.10 | NO_DRIFT |
| 0.10 – 0.20 | MINOR |
| 0.20 – 0.25 | WARNING |
| ≥ 0.25 | **CRITICAL** → triggers auto-retrain |

On CRITICAL drift, `detect_and_maybe_retrain` dispatches the `run_full_training_pipeline` Celery task by name.

### Alerting

Located in `src/neurosynth/monitoring/alerting.py`:
- **Slack**: Incoming webhook (configurable via env)
- **PagerDuty**: Events API v2
- **Structured log**: Always-on fallback

### Grafana Dashboard

`infrastructure/grafana/dashboards/neurosynth.json` — 10 panels covering inference latency, drift severity, AUC gauges, and gate status. Access at `http://localhost:3001` (admin / neurosynth).

---

## Validation & Gates

Located in `src/neurosynth/validation/`. The gate system produces one of three decisions: `PROMOTE`, `REJECT`, or `HUMAN_REVIEW`.

### Hard Gates (all must pass for PROMOTE)

| Gate | Threshold |
|---|---|
| AUC | ≥ **0.92** |
| Fairness (Equalized Odds Ratio) | 0.80 – 1.25 |
| Robustness (no critical failures) | Zero critical robustness failures |

### Soft Gates (logged, advisory)

| Gate | Threshold |
|---|---|
| ECE (calibration) | ≤ 0.05 |
| SHAP Jaccard stability | ≥ 0.60 |
| Robustness AUC drop | ≤ 0.03 |

### Validation Modules

| Module | Description |
|---|---|
| `validator.py` | AUC, F1, ECE, MCE, Brier, Youden's J, SHAP stability |
| `fairness.py` | DPR, EOR, Predictive Parity across age/sex/ethnicity |
| `fairness_postprocessor.py` | Per-group threshold equalization |
| `robustness.py` | Gaussian noise, feature dropout, covariate shift, label noise |
| `audit.py` | SHA-256 hash-chained FDA audit trail (21 CFR Part 11, IEC 62304) |
| `gates.py` | Promotion gate logic + automatic audit trail logging |
| `feature_interactions.py` | 32 → 51 feature expansion (interactions + squares) |

---

## External Data Connectors

Located in `src/neurosynth/connectors/`:

| Connector | Module | Description |
|---|---|---|
| ADNI | `adni.py` | Alzheimer's Disease Neuroimaging Initiative |
| PPMI | `ppmi.py` | Parkinson's Progression Markers Initiative |
| MIMIC | `mimic.py` | MIMIC clinical data |
| OpenNeuro | `openneuro.py` | BIDS dataset connector with NIfTI volumetric extraction (nibabel/nilearn) |
| gnomAD | `gnomad.py` | Variant frequency for 15 neurological genes via GraphQL API |
| UK Biobank | `ukbb.py` | Bulk download with field ID → feature mapping + ICD-10 filtering |
| Wearable Stream | `wearable_stream.py` | MQTT-based real-time wearable data ingestion |

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| API framework | FastAPI 0.111 + Uvicorn |
| Async DB | asyncpg 0.29 + PostgreSQL 16 |
| Task queue | Celery 5.4 + Redis 7 |
| Auth | PyJWT 2.8 (httpOnly cookie, access + refresh tokens) |
| Rate limiting | slowapi 0.1.9 |
| Structured logging | structlog 24.4 |
| Observability | prometheus-client + prometheus-fastapi-instrumentator |
| Config | pydantic-settings 2.3 |
| HTTP client | httpx 0.27 (async, with timeouts) |

### ML / Science
| Component | Technology |
|---|---|
| Ensemble | scikit-learn 1.4 (RF, GB, ET, LR) + LightGBM |
| Calibration | CalibratedClassifierCV (isotonic) |
| Conformal | MAPIE (optional) |
| SHAP | shap 0.45 |
| Deep learning | PyTorch 2.3.1 (LSTM) |
| Forecasting | pytorch-forecasting 1.1 (TFT) |
| Causal inference | DoWhy 0.13 + causal-learn 0.1 + pgmpy 0.1 |
| Graph neural net | torch-geometric 2.5 (GNN for connectome) |
| Genomics | transformers 4.40, scikit-allel, biopython, pysam |
| Neuroimaging | nibabel 5.2, nilearn 0.10, pydicom 2.4, SimpleITK 2.3 |
| Feature validation | pandera 0.20 |
| Survival analysis | lifelines 0.29 |
| LLM reports | anthropic (Claude sonnet-4-6) |
| LangChain | langchain 0.2, llama-index 0.10 |
| Privacy | opacus 1.5 (DP training) + presidio-analyzer (PII scrubbing) |
| Data pipeline | dask 2024.5, pyarrow 15, pyiceberg 0.7 |
| DVC | dvc.yaml (5-stage pipeline) |

### Frontend
| Component | Technology |
|---|---|
| Framework | React 18 + Vite 6 + TypeScript 5.5 |
| Styling | Tailwind CSS v4 |
| UI primitives | Radix UI (accordion, dialog, tabs, select, etc.) |
| Charts | Recharts 2.15 |
| Visualization | D3 7.9, Three.js 0.165, @react-three/fiber, @react-three/drei |
| Animation | Framer Motion 12.38 |
| State | Zustand 4.5 |
| Data fetching | TanStack Query 5.45 |
| Routing | React Router 7.13 |
| Storybook | 8.3 (component catalogue) |

### Infrastructure
| Component | Technology |
|---|---|
| Containerization | Docker + Docker Compose |
| Container registry | Docker Hub / GitHub CR |
| Kubernetes | Custom manifests + HPA (2–8 pods, CPU + latency scaling) |
| GPU nodes | EKS g4dn.xlarge via Terraform |
| Streaming | Kafka (KRaft mode, 3-broker MSK in prod) |
| Observability | Prometheus 2.52 + Grafana 11.0 |
| Metrics exporters | node-exporter, redis-exporter, postgres-exporter |
| IaC | Terraform (modules: gpu-nodes, kafka, EKS) |
| Model artifact store | Cloudflare R2 (S3-compatible) |
| Load testing | Locust (scripts/load_test.py) |
| Deployment | Vercel (frontend) + Render (backend) + Neon (Postgres) |

---

## Repository Layout

```
.
├── backend/                    FastAPI application
│   ├── api.py                  App factory, lifespan, middleware, router registration
│   ├── routers/                API route handlers
│   │   ├── auth.py             JWT cookie auth
│   │   ├── predictions.py      v1 sync analysis
│   │   ├── predictions_v2.py   v2 enhanced analysis (LIME, CF, CIs, circuit breaker)
│   │   ├── reports.py          v1 report endpoints
│   │   ├── reports_v2.py       SOAP + ICD-10 + FHIR + PDF
│   │   ├── patients.py         Patient CRUD + timeline
│   │   ├── biomarkers.py       SSE live stream
│   │   ├── causal.py           Causal graph endpoints
│   │   ├── features.py         Feature schema endpoint
│   │   ├── health.py           /health + /ready
│   │   ├── admin.py            Admin-only operations
│   │   └── pipelines.py        Celery pipeline kickoff
│   ├── core/
│   │   ├── config.py           pydantic-settings (all env vars)
│   │   ├── security.py         JWT, cookie names, roles, patient ID hashing
│   │   ├── rate_limit.py       slowapi + role-aware limits
│   │   ├── metrics.py          Prometheus metric definitions
│   │   └── logging.py          structlog configuration
│   ├── biomarker_model.py      BiomarkerPredictor (5-model ensemble, SHAP)
│   ├── disease_classifier.py   6-class DiseaseClassifier (real-data training)
│   ├── temporal_model.py       PyTorch LSTM trajectory model
│   ├── causal_engine.py        Causal graph + intervention engine
│   ├── model_registry.py       Model artifact loader + manifest validation
│   ├── report_generator.py     v1 deterministic report
│   ├── report_generator_v2.py  v2 SOAP + ICD-10 + FHIR + PDF (Jinja2 / WeasyPrint)
│   ├── report_generator_v3.py  v3 Claude LLM SOAP with hallucination guard
│   ├── data_pipeline.py        Dataset loading + preprocessing pipeline
│   ├── feature_schema.py       Feature metadata (labels, units, encodings)
│   ├── models.py               Core Pydantic v1 request/response models
│   ├── models_v2.py            v2 enhanced response models (22 fields, RFC 7807)
│   ├── tasks.py                Celery tasks (retry, parallel, chord aggregation)
│   ├── celery_app.py           Celery + Redis broker configuration
│   ├── db.py                   asyncpg Database wrapper
│   ├── db_schema.sql           Full PostgreSQL schema (idempotent, auto-applied)
│   ├── deps.py                 FastAPI dependency injection helpers
│   ├── wearable_simulator.py   AR(1)-driven wearable vitals generator
│   └── requirements.txt        Backend runtime dependencies
│
├── frontend/                   React clinical UI
│   ├── src/
│   │   ├── features/           Feature-level modules (auth, reports, forecast, genomics, etc.)
│   │   ├── components/         Shared components (layout, common)
│   │   ├── figma-system/       v2 visualization components
│   │   │   └── app/components/v2/
│   │   │       ├── RiskScoreGauge.tsx
│   │   │       ├── SHAPWaterfallPanel.tsx
│   │   │       ├── LIMEExplanationPanel.tsx
│   │   │       ├── CounterfactualPanel.tsx
│   │   │       ├── TrajectoryChart48.tsx
│   │   │       ├── ClinicalReportViewer.tsx
│   │   │       └── ModelPerformanceMonitor.tsx
│   │   ├── lib/
│   │   │   ├── brainAtlas.ts   SHAP → brain region mapping
│   │   │   └── featureSchema.ts Frontend mirror of feature schema
│   │   ├── hooks/              Custom React hooks
│   │   ├── state/              Zustand stores
│   │   ├── routes/             Page-level route components
│   │   ├── types/              TypeScript type definitions
│   │   └── styles.theme.css    Design token CSS variables
│   ├── package.json            npm dependencies
│   └── vite.config.ts          Vite + React plugin config
│
├── src/neurosynth/             Core library (installable Python package)
│   ├── models/
│   │   ├── calibrated_ensemble.py   5-model ensemble + meta-learner
│   │   └── model_hub.py             Multi-modal ModelHub (5 modalities)
│   ├── validation/
│   │   ├── validator.py             AUC, ECE, Brier, Youden's J, SHAP stability
│   │   ├── fairness.py              DPR / EOR demographic fairness
│   │   ├── fairness_postprocessor.py  Per-group threshold equalization
│   │   ├── robustness.py            Noise, dropout, covariate shift tests
│   │   ├── audit.py                 FDA hash-chained audit trail
│   │   ├── gates.py                 PROMOTE / REJECT / HUMAN_REVIEW logic
│   │   └── feature_interactions.py  32 → 51 feature engineering
│   ├── monitoring/
│   │   ├── drift_detector.py        PSI + KS drift with 4-tier severity
│   │   ├── alerting.py              Slack + PagerDuty alert dispatch
│   │   └── metrics.py               15 Prometheus metric definitions
│   ├── connectors/
│   │   ├── adni.py                  ADNI connector
│   │   ├── ppmi.py                  PPMI connector
│   │   ├── mimic.py                 MIMIC connector
│   │   ├── openneuro.py             OpenNeuro BIDS + NIfTI
│   │   ├── gnomad.py                gnomAD GraphQL variant frequencies
│   │   ├── ukbb.py                  UK Biobank bulk download
│   │   └── wearable_stream.py       MQTT wearable ingestion
│   ├── data/
│   │   ├── schema.py                54-feature Pandera schema (3-tier)
│   │   ├── quality.py               PSI, KS, PII scrubbing, outlier detection
│   │   └── feature_engineering.py   Multi-modal feature matrix builder
│   ├── causal/                      Causal inference modules
│   ├── connectome/                  GNN connectome models
│   ├── forecasting/                 TFT temporal forecasting
│   ├── genomic(s)/                  Genomic transformer
│   ├── dicom/                       DICOM processor
│   ├── llm/                         LLM corpus builder + RAG
│   ├── mlops/                       MLflow integration
│   ├── lakehouse/                   PyIceberg + Kafka lakehouse
│   ├── wearable/                    Wearable data models
│   └── graph/                       Graph utilities
│
├── models/                     Trained model artifacts (gitignored weights)
│   ├── model_manifest.json     MD5 + metrics + training metadata
│   ├── rf_model.pkl            RandomForest
│   ├── gb_model.pkl            GradientBoosting
│   ├── extra_trees_model.pkl   ExtraTrees
│   ├── lgbm_model.pkl          LightGBM
│   ├── lr_model.pkl            LogisticRegression
│   ├── lstm_model.pt           PyTorch LSTM
│   ├── scaler.pkl              StandardScaler
│   ├── disease_clf.pkl         6-class DiseaseClassifier
│   ├── disease_le.pkl          LabelEncoder for disease names
│   ├── disease_features.pkl    Feature list used by disease classifier
│   ├── causal_graph.npy        Causal adjacency matrix
│   ├── causal_vars.json        Causal variable names
│   ├── decision_threshold.pkl  Optimal Youden's J threshold
│   └── ensemble_v2/            CalibratedEnsemble v2 artifacts
│
├── infrastructure/
│   ├── k8s/model-server.yaml   GPU-tolerant K8s Deployment + HPA + PVC
│   ├── prometheus/prometheus.yml Scrape config (API, model server, GPU, Redis, PG)
│   └── grafana/dashboards/     10-panel Grafana dashboard JSON
│
├── scripts/
│   ├── pretrain.py             One-shot pretrain + manifest write
│   ├── release_gate.py         AUC gate enforcement (exit 2 below 0.92)
│   ├── train_orchestrator.py   Subprocess-based training coordinator
│   ├── load_test.py            Locust load test
│   ├── publish_model_artifacts.py  R2 upload script
│   └── data/
│       ├── build_realistic_synthetic.py  v4 synthetic data with clinical label function
│       ├── process_oasis.py    OASIS → 32-feature schema
│       └── merge_sources.py    Multi-source dataset merger
│
├── terraform/                  IaC — EKS GPU nodes + MSK Kafka
├── helm/                       Helm chart stubs
├── tests/                      Test suite (40+ test files)
├── .github/workflows/          CI: train-validate.yml (AUC gate enforced)
├── docker-compose.yml          Full local stack (11 services)
├── Dockerfile                  API image
├── Dockerfile.model            Model-server image (GPU-optional)
├── Procfile                    Render: api + celery-worker
├── render.yaml                 Render Blueprint (one-click full-stack deploy)
├── dvc.yaml                    5-stage DVC pipeline
├── pyproject.toml              Python package config + optional dependency groups
└── DEPLOYMENT.md               Full deployment guide (Vercel + Render + Neon)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 – 3.12 | 3.13 not yet supported (PyTorch constraint) |
| Node.js | 20+ | For frontend builds |
| npm | 9+ | Bundled with Node 20 |
| PostgreSQL | 14+ | Local or containerized |
| Redis | 6+ | Local or containerized (Upstash works) |
| Docker Desktop | Latest | For Compose workflow |

Optional for full feature set:
- `ANTHROPIC_API_KEY` — Claude LLM SOAP reports
- `R2_ACCOUNT_ID` + credentials — Cloudflare R2 model artifact store
- NVIDIA GPU — GPU-accelerated model-server container

---

## Environment Configuration

Copy the example and fill secrets:

```bash
cp .env.example .env
```

**Required variables:**

| Variable | Description | Example |
|---|---|---|
| `NEUROSYNTH_POSTGRES_DSN` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/neurosynth` |
| `NEUROSYNTH_REDIS_URL` | Redis URL | `redis://localhost:6379/0` |
| `NEUROSYNTH_JWT_SECRET` | ≥ 32-char random string | Use `openssl rand -hex 32` |
| `NEUROSYNTH_PATIENT_HASH_SECRET` | ≥ 32-char random string | Use `openssl rand -hex 32` |
| `NEUROSYNTH_AUTH_COOKIE_SECURE` | `false` locally, `true` in prod | `false` |
| `NEUROSYNTH_ALLOWED_ORIGINS` | CORS allowlist | `http://localhost:5173` |

**Optional variables:**

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Enables Claude v3 SOAP report generation |
| `ANTHROPIC_MODEL` | Default: `claude-sonnet-4-6` |
| `R2_ACCOUNT_ID` | Cloudflare R2 model artifact download on startup |
| `R2_ACCESS_KEY_ID` | R2 credentials |
| `R2_SECRET_ACCESS_KEY` | R2 credentials |
| `R2_BUCKET` | Default: `neurosynth-models` |
| `KEEPALIVE_ENABLED` | `1` = ping /health every 10 min (free-tier warm-up) |
| `NEUROSYNTH_APP_ENV` | `local` / `staging` / `prod` (prod rejects weak secrets) |

---

## Quickstart A: Docker Compose (Recommended)

The Compose file runs **11 services**: API, Celery worker, model-server, Postgres, Redis, Kafka, Prometheus, Grafana, and 3 exporters.

**1. (Optional but recommended) Generate the v4 training dataset:**

```bash
python scripts/data/build_realistic_synthetic.py \
  --n 15000 --noise 0.5 --gain 2.5 --seed 42 \
  --out data/realistic_v4.parquet
```

This gives the ensemble the full-strength training data (AUC 0.94). Without it, startup trains on the legacy CSV (~AUC 0.82).

**2. Configure environment:**

```bash
cp .env.example .env
# Edit .env — fill NEUROSYNTH_JWT_SECRET and NEUROSYNTH_PATIENT_HASH_SECRET at minimum
```

**3. Build and start:**

```bash
docker compose up --build -d
```

**4. Verify:**

```bash
curl -sS http://localhost:8000/health
curl -sS http://localhost:8000/ready
# Expect: models_loaded: true
```

**5. Access:**

| Service | URL |
|---|---|
| App (API + Frontend) | http://localhost:8000 |
| Grafana | http://localhost:3001 (admin / neurosynth) |
| Prometheus | http://localhost:9090 |
| Model server | http://localhost:8001 |
| Kafka | localhost:9092 |

---

## Quickstart B: Manual Local Runtime

**1. Install backend dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pip install -r backend/requirements.txt
```

**2. Build and copy frontend:**

```bash
cd frontend
npm install
npm run build
cd ..
rm -rf static && mkdir -p static
cp -R frontend/dist/* static/
```

**3. Initialize database:**

```bash
PGPASSWORD=postgres psql -h localhost -U postgres -d neurosynth \
  -f backend/db_schema.sql
```

**4. Start API:**

```bash
PYTHONPATH=. python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

**5. (Optional) Start Celery worker:**

```bash
PYTHONPATH=. python -m celery -A backend.celery_app:celery_app worker \
  -l info --concurrency=2
```

**6. (Optional) Run frontend dev server separately:**

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
# Opens at http://localhost:5173
```

---

## Quickstart C: One-Line Local Script

```bash
./run_local.sh
```

Installs dependencies, builds frontend, trains models if needed, and serves on `:8888`.

---

## Model Training & Caching

### Automatic Startup Behavior

On every API startup, `api.py` checks model cache validity:

1. Does `models/model_manifest.json` exist?
2. Do all required artifact files exist? (`scaler.pkl`, `rf_model.pkl`, `gb_model.pkl`, `lr_model.pkl`, `lstm_model.pt`, `causal_graph.npy`, `disease_clf.pkl`, `disease_le.pkl`, `disease_features.pkl`)
3. Does the manifest's `dataset_md5` match the current dataset file?

- ✅ All valid → load from disk cache (fast startup)
- ❌ Any check fails → run `scripts/pretrain.py` automatically in a `ThreadPoolExecutor`

### Manual Pretrain

```bash
# Train on the v4 high-fidelity dataset (recommended)
python scripts/data/build_realistic_synthetic.py \
  --n 15000 --noise 0.5 --gain 2.5 --seed 42 \
  --out data/realistic_v4.parquet

PYTHONPATH=. python scripts/pretrain.py \
  --dataset data/realistic_v4.parquet \
  --models-dir models
```

### Pretrain Outputs

| Artifact | Description |
|---|---|
| `models/model_manifest.json` | Metrics, dataset MD5, feature list, training metadata |
| `models/scaler.pkl` | StandardScaler |
| `models/rf_model.pkl` | RandomForest |
| `models/gb_model.pkl` | GradientBoosting |
| `models/lr_model.pkl` | LogisticRegression |
| `models/extra_trees_model.pkl` | ExtraTrees |
| `models/lgbm_model.pkl` | LightGBM |
| `models/lstm_model.pt` | PyTorch LSTM |
| `models/causal_graph.npy` | Causal adjacency matrix |
| `models/causal_vars.json` | Causal variable names |
| `models/disease_clf.pkl` | 6-class DiseaseClassifier |
| `models/disease_le.pkl` | Disease LabelEncoder |
| `models/disease_features.pkl` | Features used by disease classifier |
| `models/multi/` | Per-disease model directories |
| `models/ensemble_v2/` | CalibratedEnsemble v2 artifacts |

### R2 Cloud Storage (Free-tier Deploys)

On platforms like Render where model weights can't be committed, set:

```env
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=neurosynth-models
```

On startup, if `models/model_manifest.json` is missing and R2 is configured, the app downloads all artifacts from R2 before training validation.

Upload artifacts to R2:

```bash
python scripts/publish_model_artifacts.py
```

### DVC Pipeline

```bash
dvc repro   # Runs: load_data → quality_check → feature_engineering → train_classifier → pretrain
```

---

## Authentication

### Demo Credentials

| Username | Password | Role |
|---|---|---|
| `clinician@neurosynth.local` | `neurosynth` | `CLINICIAN` |
| `researcher@neurosynth.local` | `neurosynth` | `RESEARCHER` |
| `admin@neurosynth.local` | `neurosynth` | `ADMIN` |
| `clinician` | `neurosynth` | `CLINICIAN` |
| `researcher` | `neurosynth` | `RESEARCHER` |
| `admin` | `neurosynth` | `ADMIN` |

### How It Works

- Access and refresh JWTs stored in **httpOnly, SameSite=Lax cookies**
- `auth_context_middleware` decodes the access cookie on every request and sets `request.state.user`
- `require_role(Role.ADMIN)` FastAPI dependency enforces RBAC
- Role is **strictly validated** against the account — clinician credentials cannot sign in as admin
- In production (`NEUROSYNTH_APP_ENV=prod`), weak `jwt_secret` values are rejected at startup

### Cookie Lifecycle

| Cookie | Default TTL |
|---|---|
| `ns_access` | 30 minutes |
| `ns_refresh` | 7 days |

---

## API Verification Playbook

### 1. Login

```bash
curl -sS -c /tmp/ns.cookies \
    -H 'Content-Type: application/json' \
    -d '{"username":"clinician@neurosynth.local","password":"neurosynth","role":"CLINICIAN"}' \
    http://localhost:8000/auth/login | python3 -m json.tool
```

### 2. Create patient

```bash
curl -sS -b /tmp/ns.cookies \
    -H 'Content-Type: application/json' \
    -d '{"name":"README Validation Patient"}' \
    http://localhost:8000/patients/ | python3 -m json.tool
```

### 3. Run v1 full analysis

```bash
curl -sS -b /tmp/ns.cookies \
    -H 'Content-Type: application/json' \
    -d @test_payloads.json \
    http://localhost:8000/predictions/analyze | python3 -m json.tool
```

Expected signals:
- `probability` is model-driven (not fallback `0.5`)
- `trajectory` has 6 values
- `causal_graph.edges` is non-empty
- `disease_classification.predicted_disease` is present

### 4. Run v2 enhanced analysis

```bash
curl -sS -b /tmp/ns.cookies \
    -H 'Content-Type: application/json' \
    -d @test_payloads.json \
    http://localhost:8000/v2/predictions/analyze | python3 -m json.tool
```

Additional fields in response: `lime_explanation`, `counterfactuals`, `trajectory_48mo`, `confidence_intervals`, `causal_interventions`.

### 5. Generate SOAP clinical report

```bash
curl -sS -b /tmp/ns.cookies \
    -H 'Content-Type: application/json' \
    -d @test_payloads.json \
    http://localhost:8000/v2/reports/generate | python3 -m json.tool
```

### 6. Readiness probe

```bash
curl -sS http://localhost:8000/ready | python3 -m json.tool
```

Expect: `models_loaded: true`. `database` and `redis` flags reflect infrastructure availability.

### 7. Live biomarker stream

```bash
curl -N -b /tmp/ns.cookies http://localhost:8000/biomarkers/live/P-001
```

Events stream every 2 seconds with AR(1)-driven heart rate, oxygen saturation, and other vitals.

### 8. Feature schema

```bash
curl -sS http://localhost:8000/v2/features/schema | python3 -m json.tool
```

### 9. Circuit breaker status

```bash
curl -sS http://localhost:8000/v2/predictions/health
```

---

## Deployment

See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the full guide. Summary:

### Path A — Free Forever: Vercel + Render + Neon (Recommended)

| Layer | Service | Free tier |
|---|---|---|
| Frontend | Vercel | Always-on |
| Backend API | Render | Free, sleeps after ~15 min idle |
| PostgreSQL | Neon | 0.5 GB, always-on, no expiry |
| Redis (optional) | Upstash | Serverless free |

**Quick steps:**
1. Create Neon database → copy DSN
2. Create Render Web Service → build: `pip install -r backend/requirements-deploy.txt`, start: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
3. Set env vars on Render (see table in `DEPLOYMENT.md`)
4. Import repo to Vercel, root dir = `frontend/`, set `VITE_API_BASE_URL`

### Path B — One-Click Render Blueprint

```bash
# Push repo to GitHub, then:
# Render → New → Blueprint → select repo → Apply
# Sets up API + static frontend + Postgres automatically
```

> ⚠️ Render free Postgres expires after 30 days. Use Neon for persistence.

### CI/CD

`.github/workflows/train-validate.yml` runs on push:
1. Generates v4 training data
2. Trains the full ensemble
3. Enforces **AUC ≥ 0.92** gate — fails the build below threshold

---

## Testing

### Unit & Integration Tests

```bash
# Full test suite
python -m pytest -q

# Priority-ordered test files
python -m pytest tests/test_priority1_fixes.py     # Core bug fixes
python -m pytest tests/test_priority2_pipeline.py  # Data pipeline
python -m pytest tests/test_priority3_models.py    # CalibratedEnsemble + ModelHub
python -m pytest tests/test_priority4_validation.py # Validation gates + fairness
python -m pytest tests/test_priority5_inference.py  # v2 inference + LIME + CF
python -m pytest tests/test_priority6_reports.py    # v2 reports + FHIR
python -m pytest tests/test_priority8_monitoring.py # Drift detection + alerting
```

### Phase-Specific Tests

```bash
python -m pytest tests/test_phase2_connectome_gnn.py   # GNN connectome
python -m pytest tests/test_phase3_genomics_transformer.py
python -m pytest tests/test_phase4_forecasting_tft.py   # TFT temporal model
python -m pytest tests/test_phase5_causal_engine.py
```

### Integration Pipeline

```bash
python -m pytest -q tests/integration/test_pipeline_and_tasks_qa.py
```

### Frontend Build Verification

```bash
cd frontend && npm run build
# Expect: 3000+ modules, 0 errors
```

### Load Test

```bash
locust -f scripts/load_test.py --host http://localhost:8000
```

---

## Troubleshooting

### `/ready` returns `models_loaded: false`

- The pretrain script may still be running on startup (check API logs)
- Run `python scripts/pretrain.py --dataset neurological_disease_data.csv --models-dir models` manually
- Verify `models/model_manifest.json` exists and all 9 required artifact files are present

### `/ready` shows `database: false` or `redis: false`

- Check `NEUROSYNTH_POSTGRES_DSN` and `NEUROSYNTH_REDIS_URL` in `.env`
- Confirm services are listening: `pg_isready -h localhost -U postgres` / `redis-cli ping`
- In compose: services may not be healthy yet — wait ~30s after `docker compose up`

### Slow API startup

- The manifest MD5 check may be failing (dataset file changed) → triggering retraining
- To skip retrain: run `python scripts/pretrain.py` once manually, then restart API
- Ensure `data/realistic_v4.parquet` exists if you trained on it — it must be present for MD5 match

### CORS errors in browser

- `NEUROSYNTH_ALLOWED_ORIGINS` must exactly match frontend origin (scheme + host + port, no trailing slash)
- For local dev: `NEUROSYNTH_ALLOWED_ORIGINS=http://localhost:5173`
- For Vercel: `NEUROSYNTH_ALLOWED_ORIGINS=https://your-app.vercel.app`

### Frontend shows `demo mode`

- Set `VITE_API_BASE_URL=http://localhost:8000` in `frontend/.env.local` for local dev
- For Vercel: set `VITE_API_BASE_URL` in project environment variables

### Frontend shows stale UI after changes

```bash
cd frontend && npm run build && cd ..
rm -rf static && mkdir -p static && cp -R frontend/dist/* static/
```

Then hard-refresh the browser (Cmd+Shift+R on Mac).

### v2 endpoint returns 503 "circuit breaker open"

- The breaker opens after 5 consecutive inference failures with a 30-second reset window
- Check ML stack logs for the root cause
- Wait 30 seconds, then retry

### Vercel deployment fails on secret references

- Use Vercel project environment variables instead of `vercel.json` secret references
- Keep `frontend/vercel.json` minimal (just `{"rewrites": [{"source": "/(.*)", "destination": "/index.html"}]}`)

### macOS OpenMP crash during model training

- This is a known PyTorch issue with repeated fits on macOS
- LightGBM is disabled for per-disease models (`enable_lgbm=False`) to avoid it
- Cap `n_jobs` in training scripts when running locally

---

## Changelog Highlights

### v4.0.0-alpha.1 — 2026-05-24 (Latest)
- **AUC 0.9408** achieved — LightGBM added as 5th base learner with dynamic weights
- Enriched synthetic generator (`build_realistic_synthetic.py`) with nonlinear interactions and gain control
- Unified `train.py` with AUC gate enforcement (exit code 2 below 0.92)
- Parquet dataset support (`data/realistic_v4.{parquet,csv}`)
- Production DB schema: `model_versions`, `drift_events`, `audit_log`, `feature_snapshots`
- Drift auto-retrain wired end-to-end
- Anatomical brain mapping + `regionsFromShap` aggregation
- CI gate: `.github/workflows/train-validate.yml`

### v3.0.0-alpha.1 — 2026-05-23
- **LLM SOAP reports**: Claude integration with hallucination guard
- **3D brain visualization**: Three.js procedural mesh colored by SHAP
- **Feature schema**: `/v2/features/schema` + frontend FeatureLegend
- **Clinical Terminal design**: dark monospace interface
- **Realistic synthetic data**: clinically-grounded label function (LR AUC ≈ 0.87)
- **Auto-retrain**: drift → Celery retrain task dispatch
- **Free deployment**: `render.yaml` Blueprint + `DEPLOYMENT.md`

### v2.0.0-alpha.9 — 2026-05-16
- Docker Compose: model-server, Kafka, Prometheus, Grafana, exporters
- Terraform: EKS GPU node group + MSK Kafka
- Kubernetes: GPU-tolerant Deployment + HPA (2–8 pods)
- 7 new v2 React components (RiskScoreGauge, SHAPWaterfallPanel, TrajectoryChart48, etc.)
- v2 report generator: SOAP + ICD-10 + FHIR R4 + PDF
- `/v2/predictions/analyze`: LIME, counterfactuals, 48-month trajectory, circuit breaker
- PSI + KS drift detection (4-tier), Slack + PagerDuty alerting
- ECE: 0.109 → **0.020** (isotonic calibration)
- Fairness gate: EOR in [0.80, 1.25]

### v2.0.0-alpha.1 — 2026-05-09
- All 5 Celery tasks: retry logic + `chord()` parallel aggregation
- All blocking ML inference moved to `ThreadPoolExecutor`
- DiseaseClassifier: real dataset training (not synthetic `rng.normal()`)
- `torch.load(weights_only=True)` security fix
- Async report generation with `httpx.AsyncClient`
- Secret validation in prod environments

---

## Acknowledgements
**AI Assistance:** Portions of this project were developed with assistance from Claude (Anthropic). All architecture decisions, clinical design choices, data pipeline logic, and implementation were directed and reviewed by the author. AI tools were used as an amplifier per CS50's academic honesty policy, not as a replacement for original work.

---

## Security & Clinical Disclaimer

### Security Notes

- JWTs are stored in `httpOnly` cookies, not `localStorage`
- Patient IDs are HMAC-hashed before logging (via `hash_patient_id`)
- Production environments reject weak `jwt_secret` / `patient_hash_secret` at startup
- `torch.load` uses `weights_only=True` to prevent pickle-based code execution
- `audit_log` is SHA-256 hash-chained for tamper detection

### Clinical Disclaimer

NeuroSynth is a **clinical AI decision support platform** intended for research and clinical workflow assistance. It is **not** a replacement for physician judgment, medical diagnosis, or treatment planning.

Before any production clinical use:
- Conduct appropriate clinical validation on your patient population
- Perform data privacy and HIPAA/GDPR compliance reviews
- Review model governance per your regulatory jurisdiction (FDA SaMD, EU MDR, etc.)
- Assess model performance for demographic subgroups relevant to your patient panel
- Establish human oversight workflows for any model-assisted decisions

The hallucination guard in `ClinicalReportGeneratorV3` reduces LLM fabrication risk but does not eliminate it. All AI-generated reports require clinical review before use in patient care.
