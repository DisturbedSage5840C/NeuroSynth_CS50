// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
/**
 * Maps clinical feature names to the brain region most associated with them,
 * based on published AD/PD neuroscience literature. Replaces the index-based
 * SHAP→region assignment in BrainVisualization3D.tsx with anatomically grounded
 * aggregation, so the 3D brain highlights the regions a clinician would expect.
 *
 * Region ids must match REGION_LAYOUT in BrainVisualization3D.tsx.
 */

export const FEATURE_TO_REGION: Record<string, string> = {
  // Memory / medial temporal — Alzheimer's hallmarks
  MMSE: "parietal_l",
  MemoryComplaints: "temporal_l",
  Forgetfulness: "temporal_l",
  Disorientation: "parietal_r",
  Confusion: "parietal_r",

  // Executive / frontal
  FunctionalAssessment: "frontal_l",
  ADL: "frontal_r",
  DifficultyCompletingTasks: "frontal_r",
  PersonalityChanges: "frontal_l",
  BehavioralProblems: "frontal_r",

  // Affective / visual association
  Depression: "occipital_l",

  // Vascular / systemic — brainstem & cerebellum
  Hypertension: "brainstem",
  SystolicBP: "brainstem",
  DiastolicBP: "brainstem",
  CardiovascularDisease: "cerebellum",
  Diabetes: "cerebellum",
  CholesterolTotal: "cerebellum",

  _default: "parietal_l",
};

export function mapFeatureToRegion(featureName: string): string {
  return FEATURE_TO_REGION[featureName] ?? FEATURE_TO_REGION._default;
}
