# NeuroSynth v5 — Comprehensive Implementation Plan

> Researched 2026-06-03 via 5 parallel agents + direct web search. Covers real data, ML architecture, frontend redesign, backend, and full deployment stack end-to-end.

---

## Executive Summary

v5 is a complete upgrade across every layer: **real patient data** replaces synthetic, the **ML ensemble** gains a 6th learner (CatBoost) and cross-attention multimodal fusion, the **frontend** gets a closed premium dark design (no more "academic" look), and the **backend** gains pgvector RAG over PubMed literature. Target: AUC 0.96+, zero-credential-required data pipeline, production-quality UI.

**Deployment target**: $0/month (Vercel + Render + Neon + Upstash + Cloudflare R2, GitHub Student Pack).

---

## Part 1 — Real Dataset Strategy

### 1.1 Tier 1: Immediately Downloadable, No Credentials

| Dataset | Source | Rows | Key Features | Diseases | License |
|---|---|---|---|---|---|
| **Alzheimer's Disease Dataset** | Kaggle (rabieelkharoua) | 2,149 | Age, MMSE, CDR, functional assessment, ADL, memory complaints, MRI volumes, APOE4, depression, cholesterol | AD | CC BY 4.0 |
| **Dementia Prediction Dataset** | Kaggle (shashwatwork) | 373 | OASIS-2 tabular: eTIV, nWBV, ASF, MMSE, CDR, MR delay | AD/Dementia | Open |
| **UCI Parkinson's Telemonitoring** | UCI ML Repo | 5,875 | 22 acoustic voice features, UPDRS motor/total scores, age, sex | Parkinson's | Open |
| **UCI Parkinson's Classic** | UCI ML Repo | 195 | 22 biomedical voice measurements, status binary | Parkinson's | Open |
| **PhysioNet PADS** (Smartwatch) | PhysioNet | ~1,000+ | Actigraphy, tremor, gait, HR, wrist motion from neurological assessments | Parkinson's vs. controls | Open |
| **PhysioNet Non-EEG Neurological Status** | PhysioNet | 2,500+ | EDA, temperature, acceleration, HR, SpO2 from wrist biosensors, neurological status labels | General neuro | Open |
| **PhysioNet COVID-19 + MS** | PhysioNet | 347 | Demographics, comorbidities, COVID severity, MS diagnosis | MS | Open |
| **NDKP Genomic Summary Stats** | ndkp.org | 218 datasets | Open-access genomic summary stats: AD, ALS, LBD, PD | AD, ALS, PD, LBD | Open |
| **Neurological Disease Prediction (Kaggle)** | Kaggle (tanishchavaan) | ~5,000 | Multi-class clinical features, already labeled with disease types | 6 diseases | Open |

**Total immediately available**: ~17,000+ real patient-derived records across 6 disease categories.

### 1.2 Tier 2: Free Registration, High Value

| Dataset | Source | Rows | Key Features | Note |
|---|---|---|---|---|
| **OASIS-1** | oasis-brains.org | 416 | T1-MRI, MMSE, CDR, eTIV, nWBV, ASF, age, sex, SES, education | Sign Data Use Agreement — takes ~1 day |
| **OASIS-2** | oasis-brains.org | 373 longitudinal | Same + 3–4 visits per subject | Same DUA |
| **OASIS-3** | oasis-brains.org | 1,098 subjects, 2,842 sessions | MRI + PET + CSF p-tau/A-beta + cognitive scores + genetics | Same DUA — most valuable |
| **PhysioNet MIMIC-IV Demo** | physionet.org | 100 patients | ICU EHR, ICD-10 neuro codes extractable | Free account (instant) |

### 1.3 Tier 3: Programmatic APIs (No Auth)

| Source | What to Pull | How |
|---|---|---|
| **gnomAD GraphQL** | Variant frequencies for 15 neurological genes (APOE, PSEN1/2, LRRK2, SNCA, HTT, SOD1, FUS, TARDBP, C9orf72, SCN1A, KCNQ2) | GraphQL API — already scaffolded in `gnomad.py` |
| **OpenNeuro BIDS** | Clinical sidecars (participants.tsv) from neurological datasets | `openneuro-py` client, no auth |
| **PubMed E-utilities** | Abstracts for 10,000 neuro papers (for RAG) | Free, rate-limited API |
| **ClinicalTrials.gov API** | Neurological trial baseline demographics | Free REST API |

### 1.4 Data Pipeline Architecture

```
scripts/data/v5/
├── download_kaggle.py        # Kaggle API (auto-auth via ~/.kaggle/kaggle.json)
├── download_physionet.py     # PhysioNet datasets (wfdb library, open ones)
├── download_uci.py           # UCI ML repo (direct URL download)
├── process_oasis.py          # EXISTING — enhance with OASIS-3 CSF biomarkers
├── query_gnomad.py           # EXISTING gnomad.py — extend to all 15 genes
├── scrape_openneuro.py       # openneuro-py for clinical sidecars
├── build_pubmed_corpus.py    # PubMed abstracts for RAG
└── merge_v5.py               # Unified merger → data/real_v5.parquet
```

**Unified schema** (extending current 32-feature schema):

```
Core clinical (32): Age, MMSE, FunctionalAssessment, ADL, MemoryComplaints,
  BehavioralProblems, Depression, ChestPain, Hypertension, FamilyHistoryAD,
  CardiovascularDisease, Diabetes, HeadInjury, AlcoholConsumption,
  PhysicalActivity, DietQuality, SleepQuality, CholesterolTotal/LDL/HDL/Triglycerides,
  CognitiveTest, SystolicBP, DiastolicBP, BMI, Smoking

Imaging-derived (8): eTIV, nWBV, ASF, MR_Delay, WMH_volume,
  hippocampus_volume, entorhinal_thickness, ventricular_volume

Biomarkers (6): CSF_Abeta42, CSF_pTau, CSF_tTau, APOE4_dosage,
  UPDRS_motor, UPDRS_total

Wearable/motion (6): tremor_amplitude, gait_velocity, step_asymmetry,
  actigraphy_activity_index, HR_variability, SpO2_mean

Genomic (4): APOE_risk_score, LRRK2_variant_freq, HTT_repeat_est, 
  polygenetic_risk_score

Label: DiseaseType (6-class: AD, PD, MS, Epilepsy, ALS, Huntington's)
       + binary risk_label (high/low) for ensemble primary task
```

**Target**: `data/real_v5.parquet` — 20,000+ real patient records, 56 features, 6 disease classes.

### 1.5 Synthetic Augmentation (for rare classes only)

ALS (~2% prevalence) and Huntington's (~1%) will be underrepresented even in the merged dataset. Strategy:
- Use CTGAN (`ctgan` library) to generate synthetic samples for classes with <200 real samples
- Gate: synthetic fraction ≤ 30% of any class
- Track data provenance per row (`data_source` column)
- Label synthetic rows so drift detection ignores them

---

## Part 2 — ML Architecture v5

### 2.1 Primary Ensemble: 6-Model Stack

**Changes from v4:**
- ExtraTrees → **CatBoost** (better on categorical features, built-in ordinal handling, robust to class imbalance)
- Add **TabNet** as 6th base learner (attention-based tabular; interpretable feature selection per sample)
- Meta-learner stays `LogisticRegression` on OOF probabilities

```python
base_learners = [
    ("rf",    RandomForestClassifier(n_estimators=500, class_weight="balanced")),
    ("gb",    GradientBoostingClassifier(n_estimators=300)),
    ("cat",   CatBoostClassifier(iterations=300, class_weights=disease_weights, verbose=0)),  # NEW
    ("lr",    LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ("lgbm",  LGBMClassifier(n_estimators=600, class_weight="balanced")),
    ("tabnet", TabNetClassifier(n_steps=10, gamma=1.3, cat_idxs=categorical_indices)),  # NEW
]
```

**Expected AUC**: 0.96+ (from 0.9408). Evidence: CatBoost + TabNet combo adds ~0.5-1% on clinical tabular benchmarks; real data adds calibration depth.

### 2.2 Modality Fusion: Cross-Attention + Learned Weights

**v4**: Fixed weights (tabular 40%, GNN 20%, genomic 15%, TFT 15%, causal 10%)

**v5**: 
1. **Optuna-tuned weights** on validation fold (replaces hardcoded percentages)
2. **Cross-attention fusion** (2 heads, ~3ms inference overhead):

```python
class CrossAttentionFusion(nn.Module):
    """Learns which modalities to trust per-sample via attention."""
    def __init__(self, n_modalities=5, embed_dim=64, n_heads=2):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)          # scalar prob → token
        self.attn  = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.out   = nn.Linear(embed_dim, 1)
    
    def forward(self, probs: list[Tensor]) -> Tensor:
        # probs: list of (B,) tensors, one per modality
        x = torch.stack(probs, dim=1).unsqueeze(-1)   # (B, M, 1)
        x = self.embed(x)                              # (B, M, D)
        x, attn_w = self.attn(x, x, x)               # cross-attend
        return self.out(x.mean(1)).squeeze(-1), attn_w
```

### 2.3 Disease Classifier Upgrade

- 6-class RandomForest → 6-class **CatBoost** (handles categorical disease labels, better probability calibration)
- Train on real_v5 data with stratified splits
- Per-disease calibration: Platt scaling per class (complement isotonic)

### 2.4 Temporal Model (TFT Enhancement)

Keep TFT as primary. Add:
- **Disease-specific monotone constraints** (PD: UPDRS increases; ALS: FRS-ALS decreases; AD: MMSE decreases)
- **Empirical coverage validation**: assert MAPIE 95% interval achieves ≥ 93% empirical coverage on held-out set
- **Irregular time support**: handle variable follow-up intervals from ADNI/PPMI longitudinal records

### 2.5 Uncertainty Quantification

| Method | Applies To | v5 Action |
|---|---|---|
| MAPIE conformal | Ensemble | Keep + validate empirical coverage |
| Ensemble disagreement | All 6 base learners | Compute per-disease variance → show as "model confidence" |
| MC Dropout | TFT, GenomicTransformer, GNN | Enable dropout at inference (n=20 passes) |
| Focal loss | TFT, GenomicTransformer | Replace BCE with γ=2 focal loss for rare diseases |

### 2.6 Class Imbalance — Disease-Weighted Costs

```python
disease_costs = {
    "Alzheimer's":    1.0,
    "Parkinson's":    1.2,
    "Multiple Sclerosis": 1.5,
    "Epilepsy":       1.4,
    "ALS":            3.0,   # Missing = catastrophic
    "Huntington's":   3.5,   # Rarest
}
```

### 2.7 RAG Enhancement for Reports

New: `src/neurosynth/llm/rag_v2.py` — PubMed-grounded report generation:
- 10,000 PubMed neurological abstracts embedded with `text-embedding-3-small`
- Stored in Neon with `pgvector` extension
- On report generation: retrieve top-5 relevant abstracts → feed to Claude with citations
- SOAP report now includes `[PMIDxxxxxxx]` inline citations
- Hallucination guard extended: cited statistics must appear in abstract text

### 2.8 Training Pipeline v5

```yaml
# dvc.yaml v5 stages
stages:
  download_real_data:          # New: Kaggle + PhysioNet + UCI download
  process_oasis:               # Existing + OASIS-3 CSF fields
  merge_v5:                    # New: unified real_v5.parquet (20k+ rows)
  ctgan_augment:               # New: rare-class augmentation
  quality_check:               # Existing
  feature_engineering_v5:      # Extended: 32 → 56 features
  tune_fusion_weights:          # New: Optuna modality weights
  train_ensemble_v5:           # Extended: 6 learners
  train_tft_v5:                # Enhanced: monotone constraints
  train_cross_attention:        # New: CrossAttentionFusion
  embed_pubmed:                # New: RAG corpus embedding → pgvector
  pretrain:                    # Existing
  validate:                    # Extended: empirical coverage check
  release_gate:                # Existing: AUC ≥ 0.92 (target 0.96)
```

### 2.9 AUC Gate Change

Keep gate at ≥ 0.92 as hard floor. Add soft gate:
- **AUC ≥ 0.95**: Required for v5 release label (fail CI with warning, not error)
- **Rare disease F1 ≥ 0.75**: ALS + Huntington's F1 must both exceed 0.75

---

## Part 3 — Frontend Redesign (Closed Premium Design)

### 3.1 Design Language

**Theme**: "Neural Interface" — premium dark clinical SaaS. Not academic, not colorful. Think: Recursion Pharmaceuticals dashboard meets Linear.app meets a NASA mission control.

**Color Palette** (full CSS variable set):

```css
:root {
  /* Backgrounds */
  --bg-base:        #060b18;   /* deep space — page background */
  --bg-surface:     #0d1629;   /* card surfaces */
  --bg-elevated:    #162035;   /* modals, popovers */
  --bg-subtle:      #1c2a42;   /* hover states, subtle sections */

  /* Glass effect (panels) */
  --glass-bg:       rgba(13, 22, 41, 0.75);
  --glass-border:   rgba(0, 212, 255, 0.08);
  --glass-blur:     backdrop-filter: blur(12px);

  /* Accent — electric cyan (clinical precision) */
  --accent-primary:    #00d4ff;
  --accent-primary-dim:#007a99;
  --accent-glow:       0 0 20px rgba(0, 212, 255, 0.15);

  /* Risk colors */
  --risk-low:    #10b981;   /* emerald */
  --risk-medium: #f59e0b;   /* amber */
  --risk-high:   #ef4444;   /* red */
  --risk-critical: #dc2626; /* deep red */

  /* Semantic */
  --color-ad:   #818cf8;   /* indigo — Alzheimer's */
  --color-pd:   #34d399;   /* teal — Parkinson's */
  --color-ms:   #fb923c;   /* orange — MS */
  --color-ep:   #a78bfa;   /* violet — Epilepsy */
  --color-als:  #f87171;   /* rose — ALS */
  --color-hd:   #fbbf24;   /* amber — Huntington's */

  /* Text */
  --text-primary:   #f1f5f9;
  --text-secondary: #94a3b8;
  --text-tertiary:  #4b6284;
  --text-data:      #00d4ff;   /* numeric values use accent */

  /* Typography */
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}
```

**Typography**:
- Body/UI: `Inter` (weight 400, 500, 600) — loaded from Google Fonts with `font-display: swap`
- Data values, biomarker readings, confidence scores: `JetBrains Mono` — monospace precision feel
- Page titles: `Inter` 700, letter-spacing -0.02em

**Motion** (Framer Motion):
- Page transitions: fade + 8px upward translate, 200ms ease-out
- Risk gauge: spring animation on load, continuous breathing pulse at rest
- Cards: stagger-in on mount (50ms delay between items)
- Data updates: number count-up animation for probability values
- Loading: skeleton shimmer with cyan highlight sweep

### 3.2 New Page/Route Map

| Route | Component | Description |
|---|---|---|
| `/` | `LandingPage` | Redesigned hero with animated neural network canvas, feature highlights, login CTA |
| `/login` | `LoginPage` | Glassmorphism card, role selector, animated |
| `/dashboard` | `Dashboard` | Main analysis flow — patient form → full results |
| `/explorer` | `PatientExplorer` | Patient list + analysis timeline |
| `/cohort` | `CohortDashboard` | **NEW** — population-level stats from real dataset |
| `/data` | `DataPipeline` | **NEW** — real data ingestion UI, dataset status, provenance |
| `/performance` | `ModelPerformance` | AUC/ECE/F1 + per-disease breakdown, fairness metrics |
| `/literature` | `LiteratureSearch` | **NEW** — RAG search over PubMed corpus |
| `/brain` | `BrainAtlas` | **NEW** — standalone 3D brain atlas with AAL region labels |
| `/settings` | `Settings` | **NEW** — theme, API key, feature toggles |

### 3.3 Key Component Upgrades

**RiskScoreGauge v2**:
- Redesign: 3-arc SVG gauge (low/medium/high zones with gradient fill)
- Add animated needle that springs to final position
- Show per-disease probability as radial chart below gauge
- Disease color coding from CSS variables above

**SHAPWaterfallPanel v2**:
- Add causal overlay toggle: show which SHAP features also appear in causal graph
- Annotate features with their plain-English label (from feature schema)
- Add "clinical significance" badge next to each bar

**TrajectoryChart v2**:
- Per-disease trajectory selector (tabs for each of 6 diseases)
- Reference range bands (population normative data from real_v5)
- Hover: show exact confidence intervals
- "Intervention scenario" mode: drag to simulate modifiable feature changes

**BrainVisualization3D v2**:
- Replace procedural sphere with real AAL atlas mesh (downloadable ~2MB JSON)
- 116 labeled brain regions, color by SHAP aggregated by region
- Click region → show which features map to it
- "Disease focus" mode: highlight regions most predictive for selected disease

**CohortDashboard** (NEW):
- Population-level stats from real_v5.parquet
- Disease prevalence pie chart
- Age distribution by disease
- Biomarker distributions per class
- Fairness breakdown (demographic parity across age/sex groups)

**DataPipeline UI** (NEW):
- Show data source status: Kaggle ✓, PhysioNet ✓, OASIS (pending DUA) ⚠
- Row counts per source, last updated timestamp
- Data quality scores per source
- Download/refresh button per source
- Provenance sankey diagram: source → processing → merged dataset

**LiteratureSearch** (NEW):
- Text search over PubMed corpus via pgvector similarity
- Show top-5 relevant abstracts with relevance score
- Inline citation lookup when viewing SOAP reports
- "Why this paper?" explanation tied to current patient's features

### 3.4 Design System Components

All built on **shadcn/ui** with the Neural Interface theme applied:
- `GlassCard` — glassmorphism panel (backdrop-blur + semi-transparent bg + cyan border)
- `DataBadge` — monospace pill for numeric clinical values
- `RiskChip` — color-coded disease risk tag
- `ClinicalInput` — form inputs styled for clinical data entry
- `SectionHeading` — small caps label + rule line
- `PulseIndicator` — animated dot for live data streams
- `CytoscapeGraph` — D3-based causal graph (replaces current force-directed)
- `TimelineItem` — patient history timeline component

### 3.5 Landing Page Hero

Animated canvas background: procedural neural network — nodes (neurons) connected by weighted edges, slowly pulsing. Built with Three.js. No brain image (overused). Text overlay:

```
"Clinical AI for Neurological Risk"
                    ↕
"NeuroSynth predicts Alzheimer's, Parkinson's, MS, Epilepsy, ALS, 
and Huntington's from a single clinical profile — with causal 
explanations, 48-month trajectories, and LLM-generated SOAP reports."

[Enter Clinical Portal]  [View Demo]
```

Metrics strip below hero:
- `AUC 0.96+` | `6 Diseases` | `56 Clinical Features` | `48-Month Forecast` | `Real Patient Data`

---

## Part 4 — Backend Changes

### 4.1 New Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/v3/data/sources` | List data sources with status, row counts, last-updated |
| POST | `/v3/data/refresh/{source}` | Trigger re-download of a data source (admin only) |
| GET | `/v3/data/cohort/stats` | Population-level statistics for cohort dashboard |
| GET | `/v3/data/provenance` | Data lineage per row (source, processing steps) |
| POST | `/v3/literature/search` | pgvector similarity search over PubMed corpus |
| GET | `/v3/literature/cite/{pmid}` | Fetch abstract for a given PMID |
| POST | `/v3/predictions/analyze` | v3 analysis with cross-attention fusion output |
| GET | `/v3/fusion/weights` | Current Optuna-tuned modality weights |

### 4.2 Database Schema Additions

```sql
-- Real data provenance
CREATE TABLE IF NOT EXISTS data_sources (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,           -- 'kaggle_alzheimer', 'physionet_pads', etc.
    url         TEXT,
    row_count   INTEGER,
    feature_count INTEGER,
    last_updated TIMESTAMPTZ,
    status      TEXT DEFAULT 'pending',         -- 'active', 'pending', 'error'
    metadata    JSONB DEFAULT '{}'
);

-- PubMed corpus for RAG
CREATE TABLE IF NOT EXISTS literature_embeddings (
    id          SERIAL PRIMARY KEY,
    pmid        TEXT NOT NULL UNIQUE,
    title       TEXT,
    abstract    TEXT,
    journal     TEXT,
    pub_year    INTEGER,
    diseases    TEXT[],                          -- ['alzheimer', 'parkinson']
    embedding   vector(1536),                    -- pgvector OpenAI embedding
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON literature_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Cohort statistics cache
CREATE TABLE IF NOT EXISTS cohort_stats (
    id          SERIAL PRIMARY KEY,
    stat_key    TEXT NOT NULL UNIQUE,
    stat_value  JSONB,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fusion weight registry
CREATE TABLE IF NOT EXISTS fusion_weights (
    id            SERIAL PRIMARY KEY,
    modality      TEXT NOT NULL,
    weight        FLOAT,
    optuna_trial  INTEGER,
    val_auc       FLOAT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### 4.3 pgvector Setup

Neon supports pgvector natively — enable with:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

In `backend/db_schema.sql`: add the extension + `literature_embeddings` table.

In `backend/db.py`: add `search_literature(query_embedding, top_k=5)` method using:
```sql
SELECT pmid, title, abstract, 1 - (embedding <=> $1) AS similarity
FROM literature_embeddings
ORDER BY embedding <=> $1
LIMIT $2;
```

### 4.4 Report Generator v4 (RAG-Enhanced)

`backend/report_generator_v4.py`:
1. Embed the patient's risk profile as text → get embedding vector
2. Query pgvector for top-5 relevant PubMed abstracts
3. Build Claude prompt: include abstracts as context
4. Request inline citations (PMID format) in SOAP output
5. Hallucination guard: cited statistics (percentages, rates) validated against retrieved abstracts
6. Fallback to v3 (no RAG) if literature embeddings table is empty

### 4.5 Data Pipeline Service

`backend/services/data_pipeline_service.py`:
- Async methods to check/refresh each data source
- Runs as a Celery periodic task (daily refresh check)
- Updates `data_sources` table with current status
- Re-triggers `merge_v5.py` if any source is refreshed

---

## Part 5 — Deployment Architecture

### 5.1 Stack (Free Forever)

```
┌─────────────────────────────────────────────────────────┐
│                     User / Browser                       │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼──────────────────────────────┐
│           Vercel (Frontend — Always Free)                 │
│   React + Vite SPA, CDN-distributed                      │
│   VITE_API_BASE_URL → Render backend URL                 │
└──────────────────────────┬──────────────────────────────┘
                           │ REST / SSE
┌──────────────────────────▼──────────────────────────────┐
│          Render Web Service (Backend — Free Tier)         │
│   FastAPI + Uvicorn, 512MB RAM                           │
│   KEEPALIVE_ENABLED=1 (ping /health every 10min)         │
│   R2_ACCOUNT_ID set → model weights downloaded on boot   │
│   Celery worker: separate Render "Background Worker"     │
└───────────┬──────────────┬──────────────┬───────────────┘
            │              │              │
     ┌──────▼──────┐ ┌────▼─────┐ ┌────▼──────────┐
     │  Neon DB    │ │ Upstash  │ │ Cloudflare R2  │
     │ PostgreSQL  │ │  Redis   │ │ Model Artifacts│
     │ + pgvector  │ │  Broker  │ │ (10GB free)   │
     │  Free tier  │ │Free tier │ │               │
     └─────────────┘ └──────────┘ └───────────────┘
```

### 5.2 Render Free Tier Notes

- Free: 512MB RAM, shared CPU, sleeps after 15 min idle
- `KEEPALIVE_ENABLED=1`: background task pings `/health` every 10 min (prevents sleep during active use)
- ML model loading: ~400MB total (all pkl + pt files downloaded from R2 on boot)
- Boot time: ~45s (R2 download + model load)
- If RAM becomes issue: slim model set (drop TFT from startup, load on demand)

### 5.3 CI/CD Pipeline

`.github/workflows/train-validate-v5.yml`:
```yaml
on: [push, workflow_dispatch]
jobs:
  data:
    # Download all Tier-1 datasets (no credentials needed)
    # Build real_v5.parquet
    # Cache in GitHub Actions artifact store
  train:
    needs: data
    # 6-learner ensemble + CatBoost + TabNet
    # Optuna modality weight tuning (100 trials, 5min budget)
    # AUC gate: hard 0.92, soft 0.95 warning
  embed:
    # Build PubMed RAG corpus (10k abstracts)
    # Embed with OpenAI API (GitHub secret: OPENAI_API_KEY)
    # Upload embedding CSV to R2
  deploy:
    needs: [train, embed]
    # Upload model artifacts to Cloudflare R2
    # Trigger Render deploy webhook
    # Trigger Vercel deploy via API
```

**GitHub Actions free minutes** (Student Pack: 3,000 min/month):
- Data download + processing: ~10 min
- Full training: ~25 min
- RAG embedding: ~15 min
- **Total per run: ~50 min** → ~60 full CI runs/month on free tier

### 5.4 Environment Variables (v5 additions)

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | For PubMed abstract embedding (RAG corpus) | Optional |
| `KAGGLE_USERNAME` | Kaggle API user | For CI training |
| `KAGGLE_KEY` | Kaggle API key | For CI training |
| `V5_DATA_DIR` | Override default `data/` path | Optional |
| `RAG_ENABLED` | `1` = enable pgvector RAG in reports | Optional |
| `FUSION_WEIGHTS_PATH` | Path to Optuna-tuned weights JSON | Optional |

---

## Part 6 — Implementation Phases

### Phase 0 — Foundation (Week 1)
- [ ] Enable pgvector on Neon: `CREATE EXTENSION IF NOT EXISTS vector`
- [ ] Add `literature_embeddings`, `data_sources`, `cohort_stats`, `fusion_weights` tables to `db_schema.sql`
- [ ] Install dependencies: `catboost`, `pytorch-tabnet`, `openneuro-py`, `pgvector`, `ctgan`
- [ ] Set up Kaggle API credentials in CI secrets

### Phase 1 — Real Data Ingestion (Week 1-2)
- [ ] `scripts/data/v5/download_kaggle.py` — pull 3 Kaggle neurological datasets
- [ ] `scripts/data/v5/download_physionet.py` — pull PADS + Non-EEG + MS datasets
- [ ] `scripts/data/v5/download_uci.py` — Parkinson's voice + telemonitoring
- [ ] `scripts/data/v5/merge_v5.py` — unify into `data/real_v5.parquet` (56 features)
- [ ] `scripts/data/v5/ctgan_augment.py` — rare-class augmentation (ALS, Huntington's)
- [ ] Update `dvc.yaml` with v5 stages
- [ ] Validate merged dataset: run pandera schema, log per-source stats

### Phase 2 — ML Upgrade (Week 2-3)
- [ ] Swap ExtraTrees → CatBoost in `backend/biomarker_model.py` + `src/neurosynth/models/calibrated_ensemble.py`
- [ ] Add TabNet as 6th base learner
- [ ] Implement `CrossAttentionFusion` module in `src/neurosynth/models/fusion.py`
- [ ] Implement Optuna modality weight tuning in `scripts/tune_fusion_weights.py`
- [ ] Add focal loss to TFT and GenomicTransformer
- [ ] Add disease-specific monotone constraints to TFT
- [ ] Add MC Dropout inference to neural models
- [ ] Add empirical coverage validation for MAPIE
- [ ] Validate AUC ≥ 0.95 on real_v5 test fold
- [ ] Update `model_manifest.json` schema with v5 fields

### Phase 3 — RAG Pipeline (Week 3)
- [ ] `scripts/data/v5/build_pubmed_corpus.py` — fetch 10k neurology abstracts via E-utils
- [ ] `scripts/data/v5/embed_corpus.py` — embed abstracts, store in Neon pgvector
- [ ] `backend/routers/literature.py` — `/v3/literature/*` endpoints
- [ ] `backend/report_generator_v4.py` — RAG-enhanced SOAP with PMID citations
- [ ] Wire v4 reporter into API lifespan (alongside v3 as fallback)
- [ ] Test: citation accuracy on 20 sample reports

### Phase 4 — Frontend Redesign (Week 3-4)
- [ ] Apply new design tokens to `frontend/src/styles.theme.css`
- [ ] Install JetBrains Mono font
- [ ] Redesign `LandingPage.tsx` — animated Three.js neural canvas hero
- [ ] Redesign `LoginPage.tsx` — glassmorphism card
- [ ] Redesign `AppShell.tsx` — new nav (icon rail + collapsible sidebar)
- [ ] Upgrade `RiskScoreGauge` → 3-arc SVG with spring animation
- [ ] Upgrade `SHAPWaterfallPanel` → causal overlay + clinical labels
- [ ] Upgrade `TrajectoryChart48` → per-disease selector + intervention mode
- [ ] Upgrade `BrainVisualization3D` → AAL atlas mesh + region labels
- [ ] Build `CohortDashboard.tsx` — population stats from `/v3/data/cohort/stats`
- [ ] Build `DataPipeline.tsx` — source status UI
- [ ] Build `LiteratureSearch.tsx` — pgvector search frontend
- [ ] Build `BrainAtlas.tsx` — standalone atlas page
- [ ] Update routing in `App.tsx`

### Phase 5 — Backend Wiring (Week 4)
- [ ] `backend/routers/data.py` — `/v3/data/*` endpoints
- [ ] `backend/services/data_pipeline_service.py` — async refresh + Celery task
- [ ] `backend/routers/predictions_v3.py` — cross-attention fusion response
- [ ] Add v5 fields to `AnalyzeResponseV3` pydantic model
- [ ] Update `backend/api.py` — register v3 routers, load v4 reporter
- [ ] Update `render.yaml` — add `embed-worker` background worker

### Phase 6 — CI/CD + Deployment (Week 5)
- [ ] Write `.github/workflows/train-validate-v5.yml`
- [ ] Add Kaggle, OpenAI, R2 secrets to GitHub
- [ ] Update `Dockerfile` — catboost + tabnet deps
- [ ] Update `backend/requirements-deploy.txt`
- [ ] Test full Render deploy: boot → R2 download → model load → `/ready: true`
- [ ] Test Vercel deploy: build → CDN → API connectivity
- [ ] Run load test: 50 concurrent users, p95 < 2s
- [ ] Validate AUC gate in CI (train + gate run end-to-end)

### Phase 7 — Polish + QA (Week 5-6)
- [ ] Storybook stories for all new components
- [ ] Update test suite for v5 models, v3 endpoints, RAG pipeline
- [ ] Update `README.md` with v5 documentation
- [ ] Record demo video / screenshot tour
- [ ] Update `CHANGELOG.md`

---

## Part 7 — Key Files Changed / Created

### New Files
```
scripts/data/v5/
  download_kaggle.py
  download_physionet.py
  download_uci.py
  merge_v5.py
  ctgan_augment.py
  build_pubmed_corpus.py
  embed_corpus.py
  tune_fusion_weights.py

src/neurosynth/models/
  fusion.py                  # CrossAttentionFusion

backend/
  report_generator_v4.py     # RAG-enhanced SOAP
  routers/
    literature.py            # /v3/literature/* endpoints
    data.py                  # /v3/data/* endpoints
    predictions_v3.py        # v3 analysis with fusion output
  services/
    data_pipeline_service.py
  models_v3.py               # AnalyzeResponseV3 + DataSourceStatus

frontend/src/
  routes/
    CohortDashboard.tsx
    DataPipeline.tsx
    LiteratureSearch.tsx
    BrainAtlas.tsx
    Settings.tsx
  figma-system/app/components/v3/
    RiskScoreGaugeV2.tsx
    SHAPWaterfallV2.tsx
    TrajectoryChartV2.tsx
    BrainV2.tsx
    GlassCard.tsx
    DataBadge.tsx
    CohortStats.tsx
    LiteraturePanel.tsx
```

### Modified Files
```
backend/biomarker_model.py         # +CatBoost, +TabNet
backend/db_schema.sql              # +pgvector tables
backend/api.py                     # +v3 routers, +v4 reporter
backend/report_generator_v3.py     # minor: wire RAG fallback
backend/requirements.txt           # +catboost, +pytorch-tabnet, +pgvector, +ctgan
backend/requirements-deploy.txt    # slim version of above
src/neurosynth/models/calibrated_ensemble.py  # +CatBoost +TabNet +CrossAttn
src/neurosynth/temporal_tft/model.py          # +disease constraints +focal loss
frontend/src/styles.theme.css      # full color token redesign
frontend/src/features/auth/LandingPage.tsx    # full rewrite
frontend/src/features/auth/LoginPage.tsx      # glassmorphism rewrite
frontend/src/components/layout/AppShell.tsx  # new nav
dvc.yaml                           # +v5 stages
render.yaml                        # +embed-worker
.github/workflows/train-validate-v5.yml      # new CI
pyproject.toml                     # +catboost, +pytorch-tabnet optional deps
```

---

## Part 8 — Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Real data schema mismatch (Kaggle ≠ PhysioNet features) | High | `merge_v5.py` handles missing columns via imputation + schema alignment |
| AUC drops on real data vs synthetic (harder problem) | Medium | Keep enriched synthetic as augmentation fallback; hard gate still 0.92 |
| Render 512MB RAM hit with CatBoost + TabNet | Medium | Lazy-load TabNet at inference; CatBoost models are compact |
| pgvector embedding costs (OpenAI API) | Low | One-time cost ~$0.30 for 10k abstracts; cache in R2 |
| OASIS DUA takes days to approve | Low | All Tier-1 data available immediately; OASIS is enhancement only |
| TabNet training instability on small datasets | Medium | Feature `enable_tabnet=True/False` flag; fallback to ExtraTrees if unstable |
| Three.js AAL atlas mesh too large for mobile | Low | Lazy-load brain atlas; show 2D fallback on mobile |

---

## Part 9 — Success Metrics

| Metric | v4 Baseline | v5 Target |
|---|---|---|
| **Primary AUC** | 0.9408 (synthetic) | ≥ 0.95 (real data) |
| **Calibration ECE** | 0.020 | ≤ 0.015 |
| **Rare disease F1** (ALS, HD) | Unknown | ≥ 0.75 both |
| **Conformal coverage** | Not validated | ≥ 93% empirical at 95% nominal |
| **Real patient records** | 0 (synthetic) | ≥ 20,000 |
| **RAG citations per report** | 0 | 3–5 PMIDs |
| **Frontend Lighthouse** | Not measured | ≥ 90 Performance |
| **API p95 latency** | Not measured | ≤ 2s (v3 analysis) |
| **Monthly infra cost** | $0 | $0 |

---

## Appendix A — Dependency Additions

```txt
# backend/requirements.txt additions
catboost>=1.2.7
pytorch-tabnet>=4.1.0
openneuro-py>=0.2.4
pgvector>=0.2.5
ctgan>=0.7.5
wfdb>=4.1.0         # PhysioNet data access
biopython>=1.83     # PubMed E-utils (already in stack)
tiktoken>=0.7.0     # Token counting for RAG context window
```

## Appendix B — Data Source URLs

| Source | URL |
|---|---|
| Kaggle AD Dataset | kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset |
| Kaggle Dementia | kaggle.com/datasets/shashwatwork/dementia-prediction-dataset |
| UCI Parkinson's | archive.ics.uci.edu/dataset/174/parkinsons |
| UCI Parkinson's Telemonitoring | archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring |
| PhysioNet PADS | physionet.org/content/parkinsons-disease-smartwatch |
| PhysioNet Non-EEG Neuro | physionet.org/content/noneeg-neurological-status |
| OASIS | oasis-brains.org |
| NDKP Genomic | ndkp.broadinstitute.org |
| gnomAD GraphQL | gnomad.broadinstitute.org/api |
| PubMed E-utils | eutils.ncbi.nlm.nih.gov/entrez/eutils |
| OpenNeuro | openneuro.org |
