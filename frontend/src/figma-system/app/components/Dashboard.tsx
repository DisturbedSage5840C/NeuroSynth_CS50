// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { lazy, Suspense, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { patients } from '../data/mock-data';
import { ForecastChart } from './ForecastChart';
import { ConnectomeGraph } from './ConnectomeGraph';
import { GenomicHeatmap } from './GenomicHeatmap';
import { BiomarkerStrip } from './BiomarkerStrip';
import { RiskBadge } from './UncertaintyBadge';
import { Calendar, MapPin, Stethoscope, Brain, Copy } from 'lucide-react';
import { PatientInputPanel } from './PatientInputPanel';
import { useAnalysisStore } from '../../../state/analysisStore';
import type { AnalysisResult } from '../types/analysis';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { RiskScoreGauge, SHAPWaterfallPanel, CounterfactualPanel, ClinicalReportViewer, TrajectoryChart48, LIMEExplanationPanel, ModelPerformanceMonitor } from './v2';
import { usePatients } from '../hooks/usePatients';
import { apiFetch } from '../../../lib/api';

// Lazy: keeps Three.js out of the main bundle until an analysis renders the brain.
const BrainVisualization3D = lazy(() => import('./v2/BrainVisualization3D'));

interface DashboardProps {
  selectedPatientId: string;
}

export function Dashboard({ selectedPatientId }: DashboardProps) {
  const { data: livePatients = [] } = usePatients();
  const analysisResult = useAnalysisStore((s) => s.result);
  const setResult = useAnalysisStore((s) => s.setResult);
  const { data: modelPerf } = useQuery({
    queryKey: ['modelPerformance'],
    queryFn: () => apiFetch<{ roc_auc?: number; f1_weighted?: number; accuracy?: number }>('/predictions/model/performance'),
    staleTime: 5 * 60 * 1000,
  });
  const allPatients = livePatients.length ? livePatients : patients;
  const patient = allPatients.find((p) => p.id === selectedPatientId) || allPatients[0];
  const probability = analysisResult?.probability ?? patient.deteriorationProb;
  const riskLevel = useMemo(() => {
    if (!analysisResult) return patient.riskLevel;
    const level = String(analysisResult.risk_level || '').toLowerCase();
    if (level.includes('critical')) return 'critical';
    if (level.includes('high')) return 'high';
    if (level.includes('moderate')) return 'moderate';
    return 'low';
  }, [analysisResult, patient.riskLevel]);

  const disease = analysisResult?.disease_classification;
  const diseaseRows = Object.entries(disease?.disease_probabilities || {})
    .map(([name, prob]) => ({ name, probability: Number(prob) }))
    .sort((a, b) => b.probability - a.probability);

  const diseaseColor = (name: string) => {
    if (name.includes('Alzheimer')) return 'var(--risk-critical)';
    if (name.includes('Parkinson')) return 'var(--risk-high)';
    if (name.includes('Multiple Sclerosis')) return 'var(--risk-moderate)';
    if (name.includes('Epilepsy')) return 'var(--chart-3)';
    if (name.includes('ALS')) return 'var(--risk-critical)';
    if (name.includes('Huntington')) return 'var(--chart-4)';
    return 'var(--primary)';
  };

  const copyResults = () => {
    if (!analysisResult) return;
    const text = [
      'NeuroSynth Analysis Report',
      `Patient: ${patient.name} (${patient.mrn})`,
      `Risk: ${Math.round(analysisResult.probability * 100)}% - ${analysisResult.risk_level}`,
      `Confidence: ${analysisResult.confidence}`,
      `Disease: ${analysisResult.disease_classification?.predicted_disease || 'N/A'}`,
      `Top factors: ${analysisResult.top_risk_factors?.slice(0, 3).join(', ')}`,
      `Generated: ${new Date().toLocaleString()}`,
    ].join('\n');
    void navigator.clipboard.writeText(text);
  };

  return (
    <div className="flex flex-1 overflow-hidden">
      <aside className="w-96 border-r border-border bg-card/40">
        <PatientInputPanel onResult={setResult} patientId={patient.id} />
      </aside>

      <div className="flex-1 overflow-y-auto p-6">
        {/* Patient header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 style={{ fontSize: '20px' }} className="text-foreground">{patient.name}</h1>
              <RiskBadge level={riskLevel} />
            </div>
            <div className="mt-1 flex items-center gap-4 text-muted-foreground" style={{ fontSize: '12px' }}>
              <span className="font-mono">{patient.mrn}</span>
              <span>{patient.age}{patient.sex} · {patient.diagnosis}</span>
              <span className="flex items-center gap-1"><MapPin size={11} /> {patient.ward}</span>
              <span className="flex items-center gap-1"><Stethoscope size={11} /> {patient.attendingPhysician}</span>
              <span className="flex items-center gap-1"><Calendar size={11} /> Admitted {patient.admissionDate}</span>
            </div>
          </div>
        </div>

        {/* Main grid */}
        <div className="space-y-4">
          <div
            className="rounded-xl border p-5 mb-4 flex items-center justify-between"
            style={{
              borderColor: `var(--risk-${riskLevel})`,
              background: `var(--risk-${riskLevel}-bg)`,
            }}
          >
            <div>
              <div className="text-xs font-mono tracking-widest mb-1" style={{ color: `var(--risk-${riskLevel})` }}>
                DETERIORATION RISK - {riskLevel.toUpperCase()}
              </div>
              <div
                className="font-mono font-bold leading-none"
                style={{ fontSize: 52, color: `var(--risk-${riskLevel})` }}
              >
                {Math.round(probability * 100)}%
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {analysisResult
                  ? `Confidence: ${analysisResult.confidence} - ${analysisResult.individual_model_probs ? Object.keys(analysisResult.individual_model_probs).length : 4} models in ensemble`
                  : 'Run analysis to generate risk assessment'}
              </div>
            </div>
            <div className="flex flex-col gap-2 items-end">
              {analysisResult?.top_risk_factors?.slice(0, 3).map((f) => (
                <span key={f} className="text-xs px-2 py-1 rounded-md font-mono"
                  style={{ background: 'rgba(0,0,0,0.15)', color: `var(--risk-${riskLevel})` }}>
                  {f}
                </span>
              ))}
            </div>
          </div>

          {analysisResult && (
            <button
              onClick={copyResults}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded border border-border hover:border-border-strong"
            >
              <Copy size={12} />
              Copy results
            </button>
          )}

          {/* Row 1: Forecast */}
          <ForecastChart analysisResult={analysisResult as AnalysisResult | null} />

          {/* Row 2: Biomarker strip */}
          <BiomarkerStrip patientId={patient.id} />

          {/* Row 3: Connectome + Genomic */}
          <div className="grid grid-cols-2 gap-4">
            <ConnectomeGraph analysisResult={analysisResult as AnalysisResult | null} />
            <GenomicHeatmap analysisResult={analysisResult as AnalysisResult | null} />
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-medium text-foreground">Disease Profile</h3>
              <span className="rounded px-2 py-0.5 text-xs" style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}>
                {disease?.confidence || 'N/A'} confidence
              </span>
            </div>

            {diseaseRows.length ? (
              <>
                <div className="mb-3">
                  <div
                    className="text-lg font-semibold"
                    style={{ color: disease?.predicted_disease ? diseaseColor(disease.predicted_disease) : 'var(--muted-foreground)' }}
                  >
                    {disease?.predicted_disease}
                  </div>
                </div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={diseaseRows} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="name" tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={70} />
                  <YAxis tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip formatter={(value: number) => `${Math.round(value * 100)}%`} />
                  <Bar dataKey="probability" radius={[4, 4, 0, 0]} fill="var(--primary)" />
                </BarChart>
              </ResponsiveContainer>
              </>
            ) : (
              <div className="flex h-[220px] flex-col items-center justify-center gap-3">
                <div className="w-12 h-12 rounded-full border-2 border-dashed border-border flex items-center justify-center">
                  <Brain size={20} className="text-muted-foreground" />
                </div>
                <div className="text-center">
                  <div className="text-sm font-medium text-foreground">No disease profile yet</div>
                  <div className="text-xs text-muted-foreground mt-0.5">Run analysis to classify disease type</div>
                </div>
              </div>
            )}
          </div>

          {/* v2: Explainability & Report Panels */}
          {analysisResult && (
            <>
              {/* Risk Gauge + SHAP side-by-side */}
              <div className="grid grid-cols-3 gap-4">
                <div className="rounded-xl border border-border bg-card p-5 flex items-center justify-center">
                  <RiskScoreGauge
                    probability={probability}
                    riskLevel={riskLevel}
                    confidence={analysisResult.confidence || 'Medium'}
                    modelCount={analysisResult.individual_model_probs ? Object.keys(analysisResult.individual_model_probs).length : 5}
                  />
                </div>
                <div className="col-span-2">
                  <SHAPWaterfallPanel
                    shapValues={(analysisResult.shap_values || []).map((sv: any) => ({
                      feature: sv.feature || '',
                      value: typeof sv.value === 'number' ? sv.value : 0,
                    }))}
                  />
                </div>
              </div>

              {/* 3D Brain (SHAP-colored) */}
              <div className="rounded-xl border border-border bg-card p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-foreground">3D Brain — SHAP attribution</h3>
                  <span className="text-xs text-muted-foreground font-mono">drag to rotate · hover a region</span>
                </div>
                <Suspense
                  fallback={
                    <div className="h-[360px] flex items-center justify-center text-xs text-muted-foreground">
                      Loading 3D brain…
                    </div>
                  }
                >
                  <BrainVisualization3D
                    height={360}
                    shapValues={(analysisResult.shap_values || []).map((sv: any) => ({
                      feature: sv.feature || '',
                      value: typeof sv.value === 'number' ? sv.value : 0,
                    }))}
                  />
                </Suspense>
              </div>

              {/* 48-month Trajectory + LIME */}
              <div className="grid grid-cols-2 gap-4">
                <TrajectoryChart48
                  values={(analysisResult as any).trajectory || analysisResult.trajectory_48mo?.values || []}
                  bandsLower={analysisResult.confidence_bands?.lower || []}
                  bandsUpper={analysisResult.confidence_bands?.upper || []}
                />
                <LIMEExplanationPanel
                  limeValues={((analysisResult as any).shap_values || []).map((s: any) => ({
                    feature: s.feature,
                    weight: s.value,
                  }))}
                />
              </div>

              {/* Counterfactual + Clinical Report */}
              <div className="grid grid-cols-2 gap-4">
                <CounterfactualPanel
                  counterfactuals={(analysisResult as any).counterfactuals || []}
                />
                <ClinicalReportViewer
                  report={(analysisResult as any).report || null}
                />
              </div>

              {/* Model Performance */}
              <ModelPerformanceMonitor
                auc={modelPerf?.roc_auc}
                f1={modelPerf?.f1_weighted}
                decision={modelPerf ? (
                  (modelPerf.roc_auc ?? 0) >= 0.8 && (modelPerf.f1_weighted ?? 0) >= 0.75
                    ? 'PROMOTE' : 'HUMAN_REVIEW'
                ) : undefined}
                gates={modelPerf ? [
                  { name: 'AUC Threshold', type: 'hard', result: (modelPerf.roc_auc ?? 0) >= 0.8 ? 'PASS' : 'HARD_FAIL', metric: 'auc', value: modelPerf.roc_auc ?? 0, threshold: 0.80 },
                  { name: 'F1 Threshold', type: 'hard', result: (modelPerf.f1_weighted ?? 0) >= 0.75 ? 'PASS' : 'HARD_FAIL', metric: 'f1', value: modelPerf.f1_weighted ?? 0, threshold: 0.75 },
                ] : []}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
