// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
/**
 * Feature encoding schema (Gap 7) — bundled mirror of the backend
 * `GET /v2/features/schema` payload. Bundled so legends and SHAP labels render
 * instantly and work offline / in demo mode. `fetchFeatureSchema()` refreshes
 * from the live API when a backend is configured.
 */
import { apiFetch } from "@/lib/api";

export type FeatureType = "continuous" | "categorical" | "boolean";

export interface FeatureMeta {
  key: string;
  label: string;
  full_name: string;
  type: FeatureType;
  unit?: string;
  range?: [number, number];
  values?: Record<string, string>;
  clinical_note?: string;
}

export const FEATURE_SCHEMA: FeatureMeta[] = [
  { key: "Age", label: "Age", full_name: "Age", type: "continuous", unit: "years", range: [45, 100] },
  { key: "Gender", label: "Sex", full_name: "Biological sex", type: "categorical", values: { "0": "Female", "1": "Male", "2": "Other / Not specified" } },
  { key: "Ethnicity", label: "Ethnicity", full_name: "Ethnicity", type: "categorical", values: { "0": "White", "1": "Black / African American", "2": "Hispanic / Latino", "3": "Asian / Other" } },
  { key: "EducationLevel", label: "Education level", full_name: "Highest education level", type: "categorical", values: { "0": "None", "1": "High school", "2": "Some college", "3": "University degree+" }, clinical_note: "Lower education is associated with higher risk." },
  { key: "BMI", label: "Body mass index", full_name: "Body mass index", type: "continuous", unit: "kg/m²", range: [15, 45], clinical_note: "<18.5 underweight, >30 obese." },
  { key: "Smoking", label: "Current smoker", full_name: "Current smoker", type: "boolean", values: { "0": "No", "1": "Yes" } },
  { key: "AlcoholConsumption", label: "Alcohol consumption", full_name: "Alcohol consumption", type: "continuous", unit: "drinks/week", range: [0, 20] },
  { key: "PhysicalActivity", label: "Physical activity", full_name: "Weekly physical activity", type: "continuous", unit: "hrs/week", range: [0, 10] },
  { key: "DietQuality", label: "Diet quality", full_name: "Diet quality score", type: "continuous", unit: "score (0–10)", range: [0, 10], clinical_note: "Higher = healthier." },
  { key: "SleepQuality", label: "Sleep quality", full_name: "Sleep quality score", type: "continuous", unit: "score (0–10)", range: [0, 10], clinical_note: "Higher = better." },
  { key: "FamilyHistoryAlzheimers", label: "Family history: Alzheimer's", full_name: "Family history of Alzheimer's", type: "boolean", values: { "0": "No", "1": "Yes" }, clinical_note: "First-degree relative." },
  { key: "CardiovascularDisease", label: "Cardiovascular disease", full_name: "Cardiovascular disease", type: "boolean", values: { "0": "No", "1": "Yes" } },
  { key: "Diabetes", label: "Type 2 diabetes", full_name: "Type 2 diabetes", type: "boolean", values: { "0": "No", "1": "Yes" } },
  { key: "Depression", label: "Clinical depression", full_name: "Diagnosed clinical depression", type: "boolean", values: { "0": "No", "1": "Yes" } },
  { key: "HeadInjury", label: "Head injury history", full_name: "History of head injury", type: "boolean", values: { "0": "No", "1": "Yes" }, clinical_note: "TBI / concussion." },
  { key: "Hypertension", label: "Hypertension", full_name: "Hypertension", type: "boolean", values: { "0": "No", "1": "Yes" }, clinical_note: "Diagnosed or medicated." },
  { key: "SystolicBP", label: "Systolic blood pressure", full_name: "Systolic blood pressure", type: "continuous", unit: "mmHg", range: [80, 220], clinical_note: "Normal <120." },
  { key: "DiastolicBP", label: "Diastolic blood pressure", full_name: "Diastolic blood pressure", type: "continuous", unit: "mmHg", range: [40, 140], clinical_note: "Normal <80." },
  { key: "CholesterolTotal", label: "Total cholesterol", full_name: "Total cholesterol", type: "continuous", unit: "mg/dL", range: [100, 400], clinical_note: "Desirable <200." },
  { key: "CholesterolLDL", label: "LDL cholesterol", full_name: "LDL cholesterol", type: "continuous", unit: "mg/dL", range: [40, 300], clinical_note: "Optimal <100." },
  { key: "CholesterolHDL", label: "HDL cholesterol", full_name: "HDL cholesterol", type: "continuous", unit: "mg/dL", range: [20, 120], clinical_note: "High = protective." },
  { key: "CholesterolTriglycerides", label: "Triglycerides", full_name: "Triglycerides", type: "continuous", unit: "mg/dL", range: [40, 500], clinical_note: "Normal <150." },
  { key: "MMSE", label: "MMSE score", full_name: "Mini-Mental State Examination", type: "continuous", unit: "points (0–30)", range: [0, 30], clinical_note: "Cognitive screening. <24 suggests impairment." },
  { key: "FunctionalAssessment", label: "Functional assessment", full_name: "Functional assessment", type: "continuous", unit: "score (0–10)", range: [0, 10], clinical_note: "Higher = more independent." },
  { key: "MemoryComplaints", label: "Memory complaints", full_name: "Memory complaints", type: "boolean", values: { "0": "No", "1": "Yes" }, clinical_note: "Self or carer reported." },
  { key: "BehavioralProblems", label: "Behavioral problems", full_name: "Behavioral problems", type: "boolean", values: { "0": "No", "1": "Yes" } },
  { key: "ADL", label: "Activities of daily living", full_name: "Activities of daily living", type: "continuous", unit: "score (0–10)", range: [0, 10], clinical_note: "Higher = more independent." },
  { key: "Confusion", label: "Confusion episodes", full_name: "Confusion episodes", type: "boolean", values: { "0": "No", "1": "Yes" } },
  { key: "Disorientation", label: "Disorientation", full_name: "Disorientation", type: "boolean", values: { "0": "No", "1": "Yes" }, clinical_note: "Person, place, or time." },
  { key: "PersonalityChanges", label: "Personality changes", full_name: "Personality changes", type: "boolean", values: { "0": "No", "1": "Yes" }, clinical_note: "Compared to baseline." },
  { key: "DifficultyCompletingTasks", label: "Difficulty with tasks", full_name: "Difficulty completing tasks", type: "boolean", values: { "0": "No", "1": "Yes" }, clinical_note: "Previously routine tasks." },
  { key: "Forgetfulness", label: "Forgetfulness", full_name: "Forgetfulness", type: "boolean", values: { "0": "No", "1": "Yes" }, clinical_note: "Beyond normal aging." },
];

export const FEATURE_MAP: Record<string, FeatureMeta> = Object.fromEntries(
  FEATURE_SCHEMA.map((f) => [f.key, f]),
);

/** Human-readable label for a raw feature key (adds a simple unit when continuous). */
export function featureLabel(key: string): string {
  const meta = FEATURE_MAP[key];
  if (!meta) return key;
  if (meta.type === "continuous" && meta.unit && !meta.unit.includes("(")) {
    return `${meta.label} (${meta.unit})`;
  }
  return meta.label;
}

/** Short human-readable encoding hint, e.g. "0 = Female · 1 = Male · 2 = Other". */
export function encodingHint(key: string): string | null {
  const meta = FEATURE_MAP[key];
  if (!meta?.values) return null;
  return Object.entries(meta.values).map(([k, v]) => `${k} = ${v}`).join(" · ");
}

interface FeatureSchemaResponse {
  version: string;
  count: number;
  fields: FeatureMeta[];
}

/** Fetch the schema from the live API, falling back to the bundled copy. */
export async function fetchFeatureSchema(): Promise<FeatureMeta[]> {
  try {
    const data = await apiFetch<FeatureSchemaResponse>("/v2/features/schema");
    if (data?.fields?.length) return data.fields;
  } catch {
    /* fall through to bundled */
  }
  return FEATURE_SCHEMA;
}
