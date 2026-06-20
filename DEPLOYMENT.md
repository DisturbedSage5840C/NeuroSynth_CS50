# NeuroSynth v4 — Free End-to-End Deployment

Full stack, free: **frontend + FastAPI backend + Postgres**. Two supported paths.
The recommended path uses best-in-class free tiers that don't expire.

> The slim API runtime (`backend/requirements-deploy.txt`) omits `torch`/`shap` so it
> fits free build/disk limits. The app degrades gracefully: auth, the feature-schema
> endpoint, and clinical reports (LLM or template) all work; heavy multi-modal ML
> inference requires the Docker image on a paid host. The frontend ships a built-in
> demo mode, so it is fully interactive even before the backend is wired up.

---

## Path A — Recommended (free forever): Vercel + Render + Neon

| Layer | Service | Free tier |
|------|---------|-----------|
| Frontend | **Vercel** | free, always-on |
| Backend (FastAPI) | **Render** web service | free, sleeps after ~15 min idle |
| Postgres | **Neon** | free 0.5 GB, always-on, no expiry |
| Redis (optional) | **Upstash** | free, serverless |

### 1. Database — Neon
1. Create a project at https://neon.tech → copy the connection string.
2. Ensure it looks like `postgresql://user:pass@host/db?sslmode=require`.

### 2. Redis (optional) — Upstash
1. Create a database at https://upstash.com → copy the `redis://...` URL.
2. Skip to use the app without rate-limiting/caching.

### 3. Backend — Render
1. New → **Web Service** → connect this repo.
2. Build: `pip install -r backend/requirements-deploy.txt`
   Start: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
3. Environment:
   - `PYTHON_VERSION=3.12.7`
   - `NEUROSYNTH_APP_ENV=prod`
   - `NEUROSYNTH_POSTGRES_DSN=<Neon URL>`
   - `NEUROSYNTH_JWT_SECRET=<random 32+ chars>`
   - `NEUROSYNTH_PATIENT_HASH_SECRET=<random 32+ chars>`
   - `NEUROSYNTH_AUTH_COOKIE_SECURE=true`
   - `NEUROSYNTH_REDIS_URL=<Upstash URL>` (optional)
   - `ANTHROPIC_API_KEY=<sk-ant-...>` (optional — enables Claude SOAP reports)
   - `NEUROSYNTH_ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app`
4. Deploy. The schema is auto-applied on first boot. Verify `GET /health` → `{"status":"ok"}`.

### 4. Frontend — Vercel
1. Import the repo, set **Root Directory** to `frontend/`.
2. Framework preset: Vite (build `npm run build`, output `dist`).
3. Environment variable: `VITE_API_BASE_URL=https://<your-render-api>.onrender.com`
4. Deploy. Then add the Vercel URL to the backend's `NEUROSYNTH_ALLOWED_ORIGINS` and redeploy the backend.

---

## Path B — All-in-one one-click: Render Blueprint

`render.yaml` provisions **backend + static frontend + Postgres** on Render in one step.

1. Push this repo to GitHub.
2. Render → **New → Blueprint** → select the repo → Apply.
3. After deploy, set on the **frontend** service:
   - `VITE_API_BASE_URL=https://neurosynth-api.onrender.com` (your API URL) → redeploy.
4. Set on the **backend** service:
   - `NEUROSYNTH_ALLOWED_ORIGINS=https://neurosynth-frontend.onrender.com`
   - optionally `ANTHROPIC_API_KEY`, `NEUROSYNTH_REDIS_URL`.

> ⚠️ Render's **free Postgres expires 30 days** after creation. For permanence, delete
> the `databases:` block in `render.yaml` and point `NEUROSYNTH_POSTGRES_DSN` at Neon.

---

## Local full stack (all ML models, real inference)

```bash
cp .env.example .env          # fill secrets; add ANTHROPIC_API_KEY for live reports

# Generate the v4 training set first so the container trains the AUC-0.94 ensemble
# (otherwise startup falls back to the legacy CSV at ~0.82). docker-compose mounts
# ./data into the container, so this file is picked up automatically.
python scripts/data/build_realistic_synthetic.py --n 15000 --noise 0.5 --gain 2.5 --seed 42 --out data/realistic_v4.parquet

docker-compose up --build     # api :8000 · frontend :3000 · postgres · redis · kafka · grafana
```

> Or run the whole local backend flow (deps → data → train+gate → serve) with
> `./run_local.sh` (no Docker; serves on :8888).

## Notes & gotchas
- **Cold starts:** Render free web services sleep after ~15 min idle; the first request
  takes ~30–60s. Neon/Upstash stay warm.
- **Cross-origin auth:** the frontend calls the API with credentials. Set
  `NEUROSYNTH_ALLOWED_ORIGINS` to the exact frontend origin (no trailing slash).
- **LLM reports:** with no `ANTHROPIC_API_KEY`, reports use the deterministic SOAP
  template. With a key, Claude generates the narrative and a hallucination guard
  verifies stated risk percentages against the model output.
- **Demo mode:** if the frontend is hosted without `VITE_API_BASE_URL`, it serves
  realistic demo data so the UI is fully explorable without a backend.

---

## v5 CI/CD — GitHub Actions Secrets

The `train-validate-v5.yml` workflow self-skips any step whose secret is absent,
so you can activate features incrementally. Add secrets at:
**repo → Settings → Secrets and variables → Actions → New repository secret**

### Required for full pipeline

| Secret | Description | Where to get it |
| --- | --- | --- |
| `KAGGLE_USERNAME` | Kaggle account username | kaggle.com → Settings → API |
| `KAGGLE_KEY` | Kaggle API token | kaggle.com → Settings → API → Create New Token |
| `R2_ACCOUNT_ID` | Cloudflare account ID | Cloudflare dashboard → R2 → Overview |
| `R2_ACCESS_KEY_ID` | R2 API key ID | R2 → Manage R2 API Tokens |
| `R2_SECRET_ACCESS_KEY` | R2 API key secret | Same token creation flow |
| `R2_BUCKET` | R2 bucket name (default: `neurosynth-models`) | Create bucket in R2 console |
| `RENDER_DEPLOY_HOOK_URL` | Render redeploy webhook | Render service → Settings → Deploy Hook |
| `VERCEL_TOKEN` | Vercel deploy token | vercel.com → Settings → Tokens |
| `VERCEL_ORG_ID` | Vercel team/org ID | vercel.com → Settings → General |
| `VERCEL_PROJECT_ID` | Vercel project ID | Project → Settings → General |

### Optional (enable incrementally)

| Secret | Description | Effect when absent |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI API key | RAG corpus embedding skipped; literature search disabled |
| `NCBI_API_KEY` | NCBI E-utilities key | PubMed rate-limited to 3 req/s instead of 10 req/s |
| `VITE_API_BASE_URL` | Deployed backend URL | Frontend build uses relative URLs; smoke test skipped |
| `ANTHROPIC_API_KEY` | Claude API key | SOAP reports use deterministic template fallback |

### v5 environment variables (set on Render service)

After Render deploy, add these in the Render dashboard → Environment:

```sh
OPENAI_API_KEY=sk-...          # RAG literature search in reports
RAG_ENABLED=1                  # Enable pgvector RAG (set after embed-worker runs)
NCBI_API_KEY=...               # Optional: higher PubMed rate limit
KAGGLE_USERNAME=...            # Data refresh jobs
KAGGLE_KEY=...                 # Data refresh jobs
```

### One-time pgvector setup on Neon

After creating your Neon database, run this once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Then run `db_schema.sql` to create all tables including `literature_embeddings`.

The Render **embed-worker** service handles this automatically on first deploy once
`OPENAI_API_KEY` and `NEUROSYNTH_POSTGRES_DSN` are set.
