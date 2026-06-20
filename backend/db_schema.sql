CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    sex CHAR(1),
    mrn TEXT UNIQUE,
    diagnosis TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    patient_id TEXT REFERENCES patients(id),
    features JSONB NOT NULL,
    probability FLOAT,
    risk_level TEXT,
    confidence TEXT,
    trajectory JSONB,
    shap_values JSONB,
    causal_graph JSONB,
    report_sections JSONB,
    disease_classification JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyses_patient ON analyses(patient_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);

-- ── v4 additions ──────────────────────────────────────────────────────────

-- Registry of every trained model version (gate decisions, metrics, lineage).
CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    version TEXT NOT NULL,
    auc FLOAT,
    f1 FLOAT,
    ece FLOAT,
    brier FLOAT,
    gate_decision TEXT,                    -- PROMOTE / REJECT / HUMAN_REVIEW
    training_data_source TEXT,
    n_training_samples INTEGER,
    feature_count INTEGER,
    artifact_path TEXT,
    is_active BOOLEAN DEFAULT FALSE,
    promoted_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_version ON model_versions(version);

-- Drift events emitted by the PSI monitor, with the auto-retrain linkage.
CREATE TABLE IF NOT EXISTS drift_events (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    severity TEXT NOT NULL,               -- NO_DRIFT / MINOR / WARNING / CRITICAL
    psi_max FLOAT,
    drifted_features JSONB,
    retrain_triggered BOOLEAN DEFAULT FALSE,
    retrain_task_id TEXT
);

-- Immutable, hash-chained audit trail for FDA SaMD traceability.
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    event_type TEXT NOT NULL,             -- validation / gate_check / deployment / inference / report
    actor TEXT,                           -- user_id or 'system'
    patient_id TEXT,
    model_version TEXT,
    payload JSONB,
    sha256_chain TEXT,                    -- hash-chained for tamper evidence
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_log_patient ON audit_log(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_event ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at DESC);

-- Per-inference feature snapshots feeding the drift detector.
CREATE TABLE IF NOT EXISTS feature_snapshots (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    analysis_id TEXT REFERENCES analyses(id) ON DELETE CASCADE,
    features_json JSONB NOT NULL,
    sampled_for_drift BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_drift ON feature_snapshots(sampled_for_drift, created_at DESC);

-- Richer explainability + lineage columns on analyses.
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS lime_values JSONB;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS counterfactuals JSONB;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS confidence_intervals JSONB;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS generated_by TEXT;  -- 'claude:model' or 'jinja2-template'

-- ── v5 additions ───────────────────────────────────────────────────────────

-- pgvector extension (Neon supports this natively; no-op if already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Registry of real data source status (Kaggle, PhysioNet, OASIS, etc.)
CREATE TABLE IF NOT EXISTS data_sources (
    id            SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    url           TEXT,
    row_count     INTEGER,
    feature_count INTEGER,
    last_updated  TIMESTAMPTZ,
    status        TEXT DEFAULT 'pending',   -- 'active' | 'pending' | 'error'
    metadata      JSONB DEFAULT '{}'
);

-- PubMed abstract corpus for RAG-enhanced SOAP reports
CREATE TABLE IF NOT EXISTS literature_embeddings (
    id         SERIAL PRIMARY KEY,
    pmid       TEXT NOT NULL UNIQUE,
    title      TEXT,
    abstract   TEXT,
    journal    TEXT,
    pub_year   INTEGER,
    diseases   TEXT[],         -- e.g. ['alzheimer', 'parkinson']
    embedding  vector(1536),   -- OpenAI text-embedding-3-small
    created_at TIMESTAMPTZ DEFAULT NOW()
);
-- IVFFlat index for fast approximate cosine search
-- (created AFTER data is loaded — index on empty table is a no-op in pgvector)
CREATE INDEX IF NOT EXISTS idx_literature_embedding
    ON literature_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Cohort-level statistics cache (served to /v3/data/cohort/stats)
CREATE TABLE IF NOT EXISTS cohort_stats (
    id          SERIAL PRIMARY KEY,
    stat_key    TEXT NOT NULL UNIQUE,
    stat_value  JSONB,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Optuna-tuned modality fusion weight history
CREATE TABLE IF NOT EXISTS fusion_weights (
    id           SERIAL PRIMARY KEY,
    modality     TEXT NOT NULL,
    weight       FLOAT,
    optuna_trial INTEGER,
    val_auc      FLOAT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
