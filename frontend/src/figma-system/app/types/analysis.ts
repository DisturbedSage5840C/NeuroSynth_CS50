// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
export type AnalysisResult = {
  patient_id: string;
  prediction: number;
  probability: number;
  risk_level: string;
  confidence: string;
  individual_model_probs: Record<string, number>;
  top_risk_factors: string[];
  shap_values: { feature: string; value: number }[];
  trajectory: number[];
  confidence_bands: { lower: number[]; upper: number[] };
  causal_graph: {
    edges?: Array<{ from: string; to: string; strength: number }>;
    [key: string]: unknown;
  };
  report: {
    sections: Record<string, string>;
    raw_text?: string;
    generated_at?: string;
    word_count?: number;
  };
  disease_classification?: {
    predicted_disease?: string;
    disease_probabilities?: Record<string, number>;
    confidence?: string;
  };
};
