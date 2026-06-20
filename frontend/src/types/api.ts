// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
export interface CausalEdge {
  from: string;
  to: string;
  strength: number;
}

export interface CausalGraphResponse {
  variables: string[];
  edges: CausalEdge[];
}

export interface ForecastResponse {
  trajectory: number[];
  confidence_bands?: {
    lower?: number[];
    upper?: number[];
  };
}

export interface WearablePoint {
  timestamp: string;
  heart_rate?: number;
  eeg?: number;
  accel?: number;
}

export interface ClinicalReportPayload {
  assessment: string;
  risk_level: "critical" | "high" | "moderate" | "low";
  uncertainty_note: string;
  evidence_refs: string[];
  causal_pathways: Array<{ pathway: string; rationale: string; confidence: number }>;
  interventions: Array<{ name: string; action: string; priority: number; expected_impact: string; confidence?: number }>;
}
