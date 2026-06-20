// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { authStore } from "@/state/authStore";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";
const IS_BROWSER = typeof window !== "undefined";
const IS_HOSTED = IS_BROWSER && !["localhost", "127.0.0.1"].includes(window.location.hostname);
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true" || (IS_HOSTED && !import.meta.env.VITE_API_BASE_URL);

type DemoRole = "CLINICIAN" | "RESEARCHER" | "ADMIN";
type DemoUser = { username: string; password: string; role: DemoRole };

const DEMO_USERS: DemoUser[] = [
  { username: "clinician@neurosynth.local", password: "neurosynth", role: "CLINICIAN" },
  { username: "researcher@neurosynth.local", password: "neurosynth", role: "RESEARCHER" },
  { username: "admin@neurosynth.local", password: "neurosynth", role: "ADMIN" },
];

export interface ApiError extends Error {
  status?: number;
}

function normalizeRequestedRole(role: string): DemoRole {
  const normalized = role.trim().toUpperCase();
  if (normalized === "ADMIN") return "ADMIN";
  if (normalized === "RESEARCHER") return "RESEARCHER";
  return "CLINICIAN";
}

type DemoPatient = { patient_id: string; name: string; updated_at: string };
type DemoAnalysis = {
  id: string;
  patient_id: string;
  probability: number;
  risk_level: string;
  confidence: string;
  trajectory: number[];
  shap_values: Array<{ feature: string; value: number }>;
  disease_classification: { predicted_disease: string };
  created_at: string;
};

const DEMO_PATIENTS_KEY = "ns_demo_patients";
const DEMO_ANALYSES_KEY = "ns_demo_analyses";

function safeJson<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function getDemoPatients(): DemoPatient[] {
  if (!IS_BROWSER) return [];
  const existing = safeJson<DemoPatient[]>(localStorage.getItem(DEMO_PATIENTS_KEY), []);
  if (existing.length) return existing;
  const seeded: DemoPatient[] = [
    { patient_id: "P-001", name: "Nakamura, Kenji", updated_at: new Date().toISOString() },
    { patient_id: "P-002", name: "Okonkwo, Adaeze", updated_at: new Date().toISOString() },
  ];
  localStorage.setItem(DEMO_PATIENTS_KEY, JSON.stringify(seeded));
  return seeded;
}

function setDemoPatients(items: DemoPatient[]): void {
  if (!IS_BROWSER) return;
  localStorage.setItem(DEMO_PATIENTS_KEY, JSON.stringify(items));
}

function getDemoAnalyses(): DemoAnalysis[] {
  if (!IS_BROWSER) return [];
  return safeJson<DemoAnalysis[]>(localStorage.getItem(DEMO_ANALYSES_KEY), []);
}

function setDemoAnalyses(items: DemoAnalysis[]): void {
  if (!IS_BROWSER) return;
  localStorage.setItem(DEMO_ANALYSES_KEY, JSON.stringify(items));
}

function toRiskLevel(probability: number): string {
  if (probability >= 0.8) return "Critical";
  if (probability >= 0.6) return "High";
  if (probability >= 0.35) return "Moderate";
  return "Low";
}

function makeDemoAnalysis(patientId: string, features: Record<string, number>) {
  const age = Number(features.Age || 60);
  const mmse = Number(features.MMSE || 24);
  const forgetfulness = Number(features.Forgetfulness || 0);
  const memoryComplaints = Number(features.MemoryComplaints || 0);
  const base = 0.18 + age * 0.003 - mmse * 0.008 + (forgetfulness + memoryComplaints) * 0.08;
  const probability = Math.max(0.05, Math.min(0.93, Number(base.toFixed(4))));
  const riskLevel = toRiskLevel(probability);
  const trajectory = Array.from({ length: 6 }, (_, i) =>
    Number(Math.min(0.99, probability + i * 0.035).toFixed(4))
  );
  const shap_values = [
    { feature: "Age", value: Number((age / 100).toFixed(4)) },
    { feature: "MMSE", value: Number((-(mmse / 60)).toFixed(4)) },
    { feature: "Forgetfulness", value: Number((forgetfulness * 0.22).toFixed(4)) },
    { feature: "MemoryComplaints", value: Number((memoryComplaints * 0.18).toFixed(4)) },
    { feature: "Hypertension", value: Number(((features.Hypertension || 0) * 0.11).toFixed(4)) },
    { feature: "BMI", value: Number(((Number(features.BMI || 25) - 25) / 100).toFixed(4)) },
    { feature: "SleepQuality", value: Number((-(Number(features.SleepQuality || 5) / 100)).toFixed(4)) },
    { feature: "PhysicalActivity", value: Number((-(Number(features.PhysicalActivity || 5) / 100)).toFixed(4)) },
    { feature: "ADL", value: Number((-(Number(features.ADL || 5) / 80)).toFixed(4)) },
    { feature: "FunctionalAssessment", value: Number((-(Number(features.FunctionalAssessment || 5) / 80)).toFixed(4)) },
  ];

  const nowIso = new Date().toISOString();
  return {
    patient_id: patientId,
    prediction: probability >= 0.5 ? 1 : 0,
    probability,
    risk_level: riskLevel,
    confidence: "High",
    individual_model_probs: {
      random_forest: Number((probability - 0.03).toFixed(4)),
      gradient_boosting: Number((probability - 0.02).toFixed(4)),
      extra_trees: Number((probability + 0.01).toFixed(4)),
      logistic_regression: Number((probability + 0.02).toFixed(4)),
    },
    top_risk_factors: shap_values.slice(0, 5).map((s) => s.feature),
    shap_values,
    trajectory,
    confidence_bands: {
      lower: trajectory.map((v) => Number(Math.max(0, v - 0.08).toFixed(4))),
      upper: trajectory.map((v) => Number(Math.min(1, v + 0.08).toFixed(4))),
    },
    causal_graph: {
      edges: [
        { from: "Age", to: "Diagnosis", strength: 0.44 },
        { from: "MMSE", to: "Diagnosis", strength: 0.41 },
        { from: "Forgetfulness", to: "Diagnosis", strength: 0.37 },
      ],
    },
    report: {
      sections: {
        "1. EXECUTIVE SUMMARY": `Estimated neurological deterioration risk is ${(probability * 100).toFixed(1)}% (${riskLevel}).`,
        "2. RISK ASSESSMENT & INTERPRETATION": "Risk is influenced by cognitive and symptom burden with meaningful lifestyle modifiers.",
        "3. KEY BIOMARKER ANALYSIS": `Top features include ${shap_values.slice(0, 4).map((s) => s.feature).join(", ")}.`,
        "4. 36-MONTH PROGRESSION FORECAST": `Projected trajectory: ${trajectory.join(", ")}.`,
        "5. CAUSAL PATHWAY ANALYSIS": "Causal links indicate age and MMSE are dominant upstream factors.",
        "6. MODIFIABLE RISK FACTORS & INTERVENTIONS": "Improve sleep quality, increase physical activity, and monitor vascular risk factors.",
        "7. MONITORING PROTOCOL": "Repeat assessment every 3 months and escalate if trajectory slope increases.",
        "8. LIFESTYLE OPTIMIZATION PLAN": "Structured exercise, sleep hygiene, and cognitive engagement plan recommended.",
        "9. UNCERTAINTY & LIMITATIONS": "Outputs are AI-assisted and should be interpreted with clinician oversight.",
      },
      generated_at: nowIso,
      word_count: 170,
    },
    disease_classification: {
      predicted_disease: "Alzheimer's Disease",
      disease_probabilities: {
        "Alzheimer's Disease": 0.52,
        "Parkinson's Disease": 0.31,
        "Multiple Sclerosis": 0.05,
        Epilepsy: 0.04,
        ALS: 0.04,
        "Huntington's Disease": 0.04,
      },
      confidence: "Medium",
    },
    _saved_meta: {
      id: `A-${Date.now()}`,
      created_at: nowIso,
    },
  } as const;
}

async function demoFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method || "GET").toUpperCase();

  if ((path === "/patients" || path === "/patients/") && method === "GET") {
    return { items: getDemoPatients() } as T;
  }

  if ((path === "/patients" || path === "/patients/" || path.startsWith("/patients/?")) && method === "POST") {
    const url = new URL(path, "https://demo.local");
    let name = url.searchParams.get("name") || "";
    if (!name && typeof init.body === "string") {
      try {
        const payload = JSON.parse(init.body) as { name?: string };
        name = typeof payload.name === "string" ? payload.name : "";
      } catch {
        name = "";
      }
    }
    name = name || `New Patient ${String(Date.now()).slice(-4)}`;
    const patient_id = `P-${Math.random().toString(36).slice(2, 8).toUpperCase()}`;
    const updated_at = new Date().toISOString();
    const patients = [{ patient_id, name, updated_at }, ...getDemoPatients()];
    setDemoPatients(patients);
    return { patient_id } as T;
  }

  if (path.startsWith("/patients/") && path.endsWith("/analyses") && method === "GET") {
    const patient_id = path.split("/")[2] || "";
    const items = getDemoAnalyses()
      .filter((a) => a.patient_id === patient_id)
      .sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
    return { items } as T;
  }

  if (path === "/predictions/analyze" && method === "POST") {
    const body = typeof init.body === "string" ? JSON.parse(init.body) : (init.body as Record<string, unknown>) || {};
    const patient_id = String(body?.patient_id || "P-001");
    const features = (body?.features || {}) as Record<string, number>;
    const result = makeDemoAnalysis(patient_id, features);

    const analyses = getDemoAnalyses();
    const nextAnalysis: DemoAnalysis = {
      id: result._saved_meta.id,
      patient_id,
      probability: result.probability,
      risk_level: result.risk_level,
      confidence: result.confidence,
      trajectory: result.trajectory,
      shap_values: result.shap_values,
      disease_classification: { predicted_disease: result.disease_classification?.predicted_disease || "Alzheimer's Disease" },
      created_at: result._saved_meta.created_at,
    };
    setDemoAnalyses([nextAnalysis, ...analyses]);

    const patients = getDemoPatients();
    const idx = patients.findIndex((p) => p.patient_id === patient_id);
    if (idx >= 0) {
      patients[idx] = { ...patients[idx], updated_at: result._saved_meta.created_at };
      setDemoPatients(patients);
    }

    const { _saved_meta, ...payload } = result;
    return payload as unknown as T;
  }

  if (path === "/predictions/run" && method === "POST") {
    return {
      job_id: `job-${Date.now()}`,
      patient_id: "demo",
      queued_phases: [
        "connectome_inference",
        "genomic_risk_score",
        "temporal_forecast",
        "causal_analysis",
        "report_generation",
      ],
    } as T;
  }

  if (path === "/reports/generate" && method === "POST") {
    return {
      status: "generated",
      report: {
        sections: {
          "Clinical Summary": "Demo report generated from current in-browser analysis context.",
          "Recommendations": "Maintain regular monitoring cadence and lifestyle optimization.",
        },
        generated_at: new Date().toISOString(),
        word_count: 24,
      },
    } as T;
  }

  if (path === "/predictions/model/performance" && method === "GET") {
    return {
      accuracy: 0.943,
      f1_weighted: 0.937,
      roc_auc: 0.964,
      precision: 0.931,
      confusion_matrix: [[184, 12], [15, 163]],
    } as T;
  }

  if (path === "/predictions/model/feature_importance" && method === "GET") {
    return {
      MMSE: 0.214,
      Age: 0.182,
      ADL: 0.146,
      FunctionalAssessment: 0.139,
      Forgetfulness: 0.103,
      MemoryComplaints: 0.092,
      SleepQuality: 0.058,
      PhysicalActivity: 0.041,
      Hypertension: 0.025,
    } as T;
  }

  throw new Error(`Demo API route not implemented for ${method} ${path}`);
}

async function liveFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = authStore.getState().accessToken;
  const headers = new Headers(init.headers || {});
  headers.set("Content-Type", headers.get("Content-Type") || "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });

  if (response.status === 401 && authStore.getState().refreshToken) {
    const refreshed = await refreshToken();
    if (refreshed) {
      return liveFetch<T>(path, init);
    }
  }

  if (!response.ok) {
    let message: string;
    if (response.status === 401) {
      message = "Session expired — please sign out and sign in again.";
    } else {
      try {
        const body = await response.json() as { detail?: string };
        message = body.detail ?? response.statusText;
      } catch {
        message = response.statusText;
      }
    }
    const err: ApiError = new Error(message);
    err.status = response.status;
    throw err;
  }

  return response.json() as Promise<T>;
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  if (!DEMO_MODE) return liveFetch<T>(path, init);
  try {
    return await liveFetch<T>(path, init);
  } catch {
    return demoFetch<T>(path, init);
  }
}

export async function login(
  username: string,
  password: string,
  role: string = "CLINICIAN"
): Promise<{ access_token: string; refresh_token: string; role: string; }> {
  if (DEMO_MODE) {
    const requestedRole = normalizeRequestedRole(role);
    const account = DEMO_USERS.find(
      (user) => user.username === username.trim().toLowerCase() && user.password === password
    );
    if (!account) {
      throw new Error("Invalid username or password");
    }
    if (account.role !== requestedRole) {
      throw new Error(`Role mismatch: this account is ${account.role.toLowerCase()}`);
    }
    return {
      access_token: "demo-access",
      refresh_token: "demo-refresh",
      role: account.role,
    };
  }
  // On localhost the backend runs with DEV_BYPASS_AUTH — skip the network call
  // entirely and authenticate locally. All subsequent API calls go through the
  // Vite proxy and the backend auto-authenticates every request.
  const isLocalDev = typeof window !== "undefined" &&
    ["localhost", "127.0.0.1"].includes(window.location.hostname);

  const VALID_LOCAL_USERS: Record<string, string> = {
    "clinician@neurosynth.local": "neurosynth",
    "researcher@neurosynth.local": "neurosynth",
    "admin@neurosynth.local": "neurosynth",
    "clinician": "neurosynth",
    "researcher": "neurosynth",
    "admin": "neurosynth",
  };

  if (isLocalDev) {
    const expectedPassword = VALID_LOCAL_USERS[username.trim().toLowerCase()];
    if (!expectedPassword || password !== expectedPassword) {
      throw new Error("Invalid credentials");
    }
    const resolvedRole = username.includes("researcher") ? "RESEARCHER"
      : username.includes("admin") ? "ADMIN"
      : "CLINICIAN";
    return { access_token: "local-dev", refresh_token: "local-dev", role: resolvedRole };
  }

  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, role }),
  });
  if (!response.ok) throw new Error("Invalid credentials");
  const data = await response.json();
  return {
    access_token: data.access_token ?? "",
    refresh_token: data.refresh_token ?? "",
    role: data.user?.role ?? role,
  };
}

export async function refreshToken(): Promise<boolean> {
  if (DEMO_MODE) return true;
  const stored = authStore.getState();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (stored.refreshToken) {
    headers["Authorization"] = `Bearer ${stored.refreshToken}`;
  }
  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: "POST",
    headers,
  });

  if (!response.ok) return false;
  const payload = await response.json();
  const access = String(payload?.access_token ?? "");
  const refresh = String(payload?.refresh_token ?? stored.refreshToken ?? "");
  const role = String(payload?.user?.role ?? stored.role ?? "CLINICIAN");
  if (!access) return false;
  authStore.getState().setTokens(access, refresh, role);
  return true;
}

export function streamUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}
