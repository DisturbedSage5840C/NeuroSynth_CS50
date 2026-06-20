// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import { Brain, Loader2 } from "lucide-react";

import { apiFetch } from "../../../lib/api";
import { Slider } from "./ui/slider";
import type { AnalysisResult } from "../types/analysis";

const PATIENT_FIELDS = [
  { key: "Age", label: "Age", min: 45, max: 100, step: 1, section: "Demographics", default: 73 },
  { key: "Gender", label: "Gender", min: 0, max: 2, step: 1, section: "Demographics", default: 1 },
  { key: "Ethnicity", label: "Ethnicity", min: 0, max: 3, step: 1, section: "Demographics", default: 1 },
  { key: "EducationLevel", label: "Education Level", min: 0, max: 3, step: 1, section: "Demographics", default: 1 },
  { key: "BMI", label: "BMI", min: 15, max: 45, step: 0.1, section: "Lifestyle", default: 27.4 },
  { key: "Smoking", label: "Smoking", min: 0, max: 1, step: 1, section: "Lifestyle", default: 0 },
  { key: "AlcoholConsumption", label: "Alcohol Consumption", min: 0, max: 20, step: 0.1, section: "Lifestyle", default: 3.2 },
  { key: "PhysicalActivity", label: "Physical Activity", min: 0, max: 10, step: 0.1, section: "Lifestyle", default: 4.8 },
  { key: "DietQuality", label: "Diet Quality", min: 0, max: 10, step: 0.1, section: "Lifestyle", default: 5.2 },
  { key: "SleepQuality", label: "Sleep Quality", min: 0, max: 10, step: 0.1, section: "Lifestyle", default: 5.0 },
  { key: "FamilyHistoryAlzheimers", label: "Family History AD", min: 0, max: 1, step: 1, section: "Medical History", default: 1 },
  { key: "CardiovascularDisease", label: "Cardiovascular Disease", min: 0, max: 1, step: 1, section: "Medical History", default: 0 },
  { key: "Diabetes", label: "Diabetes", min: 0, max: 1, step: 1, section: "Medical History", default: 0 },
  { key: "Depression", label: "Depression", min: 0, max: 1, step: 1, section: "Medical History", default: 0 },
  { key: "HeadInjury", label: "Head Injury", min: 0, max: 1, step: 1, section: "Medical History", default: 0 },
  { key: "Hypertension", label: "Hypertension", min: 0, max: 1, step: 1, section: "Medical History", default: 1 },
  { key: "SystolicBP", label: "Systolic BP", min: 80, max: 220, step: 1, section: "Clinical Measurements", default: 132 },
  { key: "DiastolicBP", label: "Diastolic BP", min: 40, max: 140, step: 1, section: "Clinical Measurements", default: 82 },
  { key: "CholesterolTotal", label: "Total Cholesterol", min: 100, max: 400, step: 1, section: "Clinical Measurements", default: 202 },
  { key: "CholesterolLDL", label: "LDL", min: 40, max: 300, step: 1, section: "Clinical Measurements", default: 124 },
  { key: "CholesterolHDL", label: "HDL", min: 20, max: 120, step: 1, section: "Clinical Measurements", default: 49 },
  { key: "CholesterolTriglycerides", label: "Triglycerides", min: 40, max: 500, step: 1, section: "Clinical Measurements", default: 166 },
  { key: "MMSE", label: "MMSE", min: 0, max: 30, step: 1, section: "Clinical Measurements", default: 24 },
  { key: "FunctionalAssessment", label: "Functional Assessment", min: 0, max: 10, step: 0.1, section: "Clinical Measurements", default: 6.2 },
  { key: "ADL", label: "ADL", min: 0, max: 10, step: 0.1, section: "Clinical Measurements", default: 6.0 },
  { key: "MemoryComplaints", label: "Memory Complaints", min: 0, max: 1, step: 1, section: "Symptoms", default: 1 },
  { key: "BehavioralProblems", label: "Behavioral Problems", min: 0, max: 1, step: 1, section: "Symptoms", default: 0 },
  { key: "Confusion", label: "Confusion", min: 0, max: 1, step: 1, section: "Symptoms", default: 0 },
  { key: "Disorientation", label: "Disorientation", min: 0, max: 1, step: 1, section: "Symptoms", default: 0 },
  { key: "PersonalityChanges", label: "Personality Changes", min: 0, max: 1, step: 1, section: "Symptoms", default: 0 },
  { key: "DifficultyCompletingTasks", label: "Difficulty Completing Tasks", min: 0, max: 1, step: 1, section: "Symptoms", default: 1 },
  { key: "Forgetfulness", label: "Forgetfulness", min: 0, max: 1, step: 1, section: "Symptoms", default: 1 },
] as const;

const SECTIONS = ["Demographics", "Lifestyle", "Medical History", "Clinical Measurements", "Symptoms"] as const;

interface PatientInputPanelProps {
  onResult: (result: AnalysisResult) => void;
  patientId: string;
}

export function PatientInputPanel({ onResult, patientId }: PatientInputPanelProps) {
  const queryClient = useQueryClient();
  const defaults = useMemo(
    () => Object.fromEntries(PATIENT_FIELDS.map((f) => [f.key, f.default])) as Record<string, number>,
    []
  );
  const [patientName, setPatientName] = useState("");
  const [fields, setFields] = useState<Record<string, number>>(defaults);
  const [error, setError] = useState("");

  const mutation = useMutation({
    mutationFn: async (features: Record<string, number>) => {
      const create = await apiFetch<{ patient_id: string }>("/patients/", {
        method: "POST",
        body: JSON.stringify({
          name: patientName.trim() || `Patient ${String(Date.now()).slice(-4)}`,
        }),
      });

      return apiFetch<AnalysisResult>("/predictions/analyze", {
        method: "POST",
        body: JSON.stringify({
          patient_id: create.patient_id || patientId,
          features,
        }),
      });
    },
    onSuccess: async (data) => {
      setError("");
      await queryClient.invalidateQueries({ queryKey: ["patients"] });
      onResult(data);
      setPatientName("");
    },
    onError: (err: unknown) => {
      if (err instanceof Error) {
        setError(err.message || "Analysis failed");
      } else {
        setError("Analysis failed");
      }
    },
  });

  const handleSubmit = () => mutation.mutate(fields);
  const isBinaryField = (min: number, max: number, step: number) => min === 0 && max === 1 && step === 1;

  return (
    <div className="h-full overflow-y-auto p-4">
      <button
        onClick={() => {
          setFields(defaults);
          setError('');
        }}
        className="mb-3 w-full rounded border border-border px-3 py-2 text-xs text-muted-foreground hover:text-foreground"
      >
        New Analysis
      </button>

      <div className="mb-4 rounded-lg border border-border bg-card p-3">
        <div className="mb-2 text-xs font-medium tracking-wider text-primary uppercase">Patient Identity</div>
        <input
          type="text"
          placeholder="Enter patient name"
          value={patientName}
          onChange={(e) => setPatientName(e.target.value)}
          className="w-full rounded-md border border-border bg-background px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
        />
      </div>

      {SECTIONS.map((section) => (
        <div key={section} className="mb-4 rounded-lg border border-border bg-card p-3">
          <div className="mb-3 flex items-center gap-2">
            <div className="w-0.5 h-4 rounded-full bg-primary" />
            <h3 className="text-xs font-medium tracking-wider text-primary uppercase">{section}</h3>
          </div>
          <div className="grid grid-cols-1 gap-3">
            {PATIENT_FIELDS.filter((f) => f.section === section).map((f) =>
              isBinaryField(f.min, f.max, f.step) ? (
                <div
                  key={f.key}
                  className="flex items-center justify-between rounded-lg border border-border bg-background px-3 py-2 cursor-pointer"
                  onClick={() => setFields((prev) => ({ ...prev, [f.key]: prev[f.key] ? 0 : 1 }))}
                >
                  <span className="text-xs text-muted-foreground">{f.label}</span>
                  <div
                    className="w-9 h-5 rounded-full transition-colors flex items-center px-0.5"
                    style={{
                      background: fields[f.key] ? 'var(--primary)' : 'var(--secondary)',
                    }}
                  >
                    <div
                      className="w-4 h-4 rounded-full bg-white shadow-sm transition-transform"
                      style={{ transform: fields[f.key] ? 'translateX(16px)' : 'translateX(0)' }}
                    />
                  </div>
                </div>
              ) : (
                <div key={f.key} className="rounded-lg border border-border bg-background px-3 py-2">
                  <div className="flex justify-between items-baseline mb-2">
                    <span className="text-xs text-muted-foreground">{f.label}</span>
                    <span
                      className="font-mono text-xs font-medium px-1.5 py-0.5 rounded"
                      style={{ background: 'var(--secondary)', color: 'var(--foreground)' }}
                    >
                      {fields[f.key]}
                    </span>
                  </div>
                  <Slider
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={[fields[f.key]]}
                    onValueChange={([v]) => setFields((prev) => ({ ...prev, [f.key]: v }))}
                    className="w-full"
                  />
                  <div className="flex justify-between mt-1">
                    <span className="text-[10px] text-muted-foreground">{f.min}</span>
                    <span className="text-[10px] text-muted-foreground">{f.max}</span>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      ))}

      {error && (
        <div
          className="mb-3 rounded border border-[var(--risk-critical)] bg-[var(--risk-critical-bg)] p-2 text-[var(--risk-critical)]"
          style={{ fontSize: "12px" }}
        >
          {error}
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={mutation.isPending}
        className="flex w-full items-center justify-center gap-2 rounded-lg py-3 font-medium transition-opacity disabled:opacity-50"
        style={{ background: "var(--primary)", color: "var(--primary-foreground)", fontSize: "13px" }}
      >
        {mutation.isPending ? (
          <>
            <Loader2 size={16} className="animate-spin" /> Analyzing...
          </>
        ) : (
          <>
            <Brain size={16} /> Analyze Patient
          </>
        )}
      </button>
    </div>
  );
}
