// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useOutletContext } from 'react-router';
import {
  Brain,
  Dna,
  FlaskConical,
  Watch,
  Stethoscope,
  Sliders,
  Play,
  RotateCcw,
} from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { apiFetch } from '../../../lib/api';
import { timelineEvents } from '../data/mock-data';

type Modality = 'all' | 'imaging' | 'genomic' | 'lab' | 'wearable' | 'clinical';

const modalityConfig: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  imaging: { icon: <Brain size={12} />, color: '#818cf8', label: 'Imaging' },
  genomic: { icon: <Dna size={12} />, color: '#34d399', label: 'Genomic' },
  lab: { icon: <FlaskConical size={12} />, color: '#f59e0b', label: 'Lab' },
  wearable: { icon: <Watch size={12} />, color: '#ec4899', label: 'Wearable' },
  clinical: { icon: <Stethoscope size={12} />, color: '#06b6d4', label: 'Clinical' },
};

const flagStyles: Record<string, string> = {
  normal: 'text-[var(--risk-low)] bg-[var(--risk-low-bg)]',
  abnormal: 'text-[var(--risk-moderate)] bg-[var(--risk-moderate-bg)]',
  critical: 'text-[var(--risk-critical)] bg-[var(--risk-critical-bg)]',
};

type HistoryItem = {
  id: string;
  probability: number;
  risk_level: string;
  confidence: string;
  trajectory: number[];
  shap_values: Array<{ feature: string; value: number }>;
  disease_classification?: { predicted_disease?: string };
  created_at: string;
};

function normalizeShapValues(input: unknown): Array<{ feature: string; value: number }> {
  const parseMaybeJson = (raw: unknown) => {
    if (typeof raw !== 'string') return raw;
    try {
      return JSON.parse(raw);
    } catch {
      return [];
    }
  };

  const parsed = parseMaybeJson(input);
  if (!Array.isArray(parsed)) return [];
  return parsed
    .map((item) => {
      if (!item || typeof item !== 'object') return null;
      const feature = String((item as { feature?: unknown }).feature ?? 'unknown');
      const value = Number((item as { value?: unknown }).value ?? 0);
      return { feature, value };
    })
    .filter((item): item is { feature: string; value: number } => item !== null);
}

export function DataExplorer() {
  const { selectedPatientId } = useOutletContext<{ selectedPatientId: string }>();
  const [filter, setFilter] = useState<Modality>('all');
  const [simOpen, setSimOpen] = useState(false);
  const [simParam, setSimParam] = useState(50);
  const [simRunning, setSimRunning] = useState(false);
  const [leftSelection, setLeftSelection] = useState<string | null>(null);
  const [rightSelection, setRightSelection] = useState<string | null>(null);

  const historyQuery = useQuery({
    queryKey: ['patient-history', selectedPatientId],
    queryFn: () => apiFetch<{ items: HistoryItem[] }>(`/patients/${selectedPatientId}/analyses`),
  });

  const historyItems = Array.isArray(historyQuery.data?.items)
    ? historyQuery.data.items
    : [];
  const chartData = historyItems
    .slice()
    .reverse()
    .map((h, i) => ({
      idx: i + 1,
      probability: Number(h.probability || 0),
      created_at: h.created_at,
      id: h.id,
    }));

  const selectedLeft = useMemo(
    () => historyItems.find((h) => h.id === leftSelection) || historyItems[0],
    [historyItems, leftSelection]
  );
  const selectedRight = useMemo(
    () => historyItems.find((h) => h.id === rightSelection) || historyItems[1] || historyItems[0],
    [historyItems, rightSelection]
  );

  const filtered = filter === 'all' ? timelineEvents : timelineEvents.filter(e => e.modality === filter);

  const runSim = () => {
    setSimRunning(true);
    setTimeout(() => setSimRunning(false), 2000);
  };

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Main timeline */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 style={{ fontSize: '20px' }} className="text-foreground">Multi-Modal Data Explorer</h1>
            <p className="text-muted-foreground mt-0.5" style={{ fontSize: '12px' }}>Patient timeline across all modalities · Nakamura, Kenji</p>
          </div>
          <button
            onClick={() => setSimOpen(!simOpen)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md border transition-colors ${simOpen ? 'bg-primary/15 border-primary text-primary' : 'border-border text-muted-foreground hover:text-foreground hover:border-border-strong'}`}
            style={{ fontSize: '12px' }}
          >
            <Sliders size={14} />
            Counterfactual Simulator
          </button>
        </div>

        {/* Modality filters */}
        <div className="flex items-center gap-2 mb-6">
          <button
            onClick={() => setFilter('all')}
            className={`px-3 py-1 rounded-md transition-colors ${filter === 'all' ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
            style={{ fontSize: '11px' }}
          >
            All ({timelineEvents.length})
          </button>
          {Object.entries(modalityConfig).map(([key, config]) => {
            const count = timelineEvents.filter(e => e.modality === key).length;
            return (
              <button
                key={key}
                onClick={() => setFilter(key as Modality)}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-colors ${filter === key ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'}`}
                style={{ fontSize: '11px', ...(filter === key ? { backgroundColor: config.color + '20', color: config.color } : {}) }}
              >
                {config.icon}
                {config.label} ({count})
              </button>
            );
          })}
        </div>

        {/* Timeline */}
        <div className="relative">
          <div className="absolute left-[18px] top-0 bottom-0 w-px bg-border" />
          <div className="space-y-1">
            {filtered.map(event => {
              const config = modalityConfig[event.modality];
              return (
                <div key={event.id} className="flex gap-4 group">
                  {/* Node */}
                  <div className="relative z-10 mt-3">
                    <div className="w-[38px] h-[38px] rounded-full border-2 flex items-center justify-center"
                      style={{ borderColor: config.color, backgroundColor: config.color + '15' }}>
                      <span style={{ color: config.color }}>{config.icon}</span>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="flex-1 bg-card border border-border rounded-lg p-3 mb-2 group-hover:border-border-strong transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-foreground" style={{ fontSize: '13px' }}>{event.title}</span>
                        {event.flag && event.flag !== 'normal' && (
                          <span className={`px-1.5 py-0.5 rounded font-mono ${flagStyles[event.flag]}`}
                            style={{ fontSize: '9px', letterSpacing: '0.05em' }}>
                            {event.flag.toUpperCase()}
                          </span>
                        )}
                      </div>
                      <span className="text-muted-foreground font-mono" style={{ fontSize: '10px' }}>{event.timestamp}</span>
                    </div>
                    <p className="text-muted-foreground mt-1" style={{ fontSize: '11px' }}>{event.description}</p>
                    {event.value && (
                      <div className="mt-1.5 font-mono text-foreground" style={{ fontSize: '12px' }}>
                        {event.value}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="mt-6 rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 text-sm font-medium text-foreground">Patient Risk Timeline</h3>
          {chartData.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="idx" tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} />
                <Tooltip
                  formatter={(value: number) => `${Math.round(value * 100)}%`}
                  labelFormatter={(v) => `Analysis #${v}`}
                />
                <Line type="monotone" dataKey="probability" stroke="var(--primary)" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-xs text-muted-foreground">No stored analyses yet for this patient.</div>
          )}

          {historyItems.length > 0 && (
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="rounded border border-border p-3">
                <h4 className="mb-2 text-xs text-muted-foreground">Select Analysis A</h4>
                <select
                  className="mb-3 w-full rounded border border-border bg-secondary px-2 py-1 text-xs"
                  value={selectedLeft?.id || ''}
                  onChange={(e) => setLeftSelection(e.target.value)}
                >
                  {historyItems.map((h) => (
                    <option key={h.id} value={h.id}>{new Date(h.created_at).toLocaleString()}</option>
                  ))}
                </select>
                <ScatterChart width={340} height={160}>
                  <CartesianGrid stroke="var(--border)" />
                  <XAxis type="number" dataKey="i" name="Rank" tick={{ fill: 'var(--muted-foreground)', fontSize: 10 }} />
                  <YAxis type="number" dataKey="v" name="SHAP" tick={{ fill: 'var(--muted-foreground)', fontSize: 10 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter
                    data={normalizeShapValues(selectedLeft?.shap_values)
                      .slice(0, 12)
                      .map((s, i) => ({ i: i + 1, v: Number(s.value || 0), feature: s.feature }))}
                    fill="var(--risk-high)"
                  />
                </ScatterChart>
              </div>

              <div className="rounded border border-border p-3">
                <h4 className="mb-2 text-xs text-muted-foreground">Select Analysis B</h4>
                <select
                  className="mb-3 w-full rounded border border-border bg-secondary px-2 py-1 text-xs"
                  value={selectedRight?.id || ''}
                  onChange={(e) => setRightSelection(e.target.value)}
                >
                  {historyItems.map((h) => (
                    <option key={h.id} value={h.id}>{new Date(h.created_at).toLocaleString()}</option>
                  ))}
                </select>
                <ScatterChart width={340} height={160}>
                  <CartesianGrid stroke="var(--border)" />
                  <XAxis type="number" dataKey="i" name="Rank" tick={{ fill: 'var(--muted-foreground)', fontSize: 10 }} />
                  <YAxis type="number" dataKey="v" name="SHAP" tick={{ fill: 'var(--muted-foreground)', fontSize: 10 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter
                    data={normalizeShapValues(selectedRight?.shap_values)
                      .slice(0, 12)
                      .map((s, i) => ({ i: i + 1, v: Number(s.value || 0), feature: s.feature }))}
                    fill="var(--risk-moderate)"
                  />
                </ScatterChart>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Counterfactual Simulator Panel */}
      {simOpen && (
        <div className="w-80 border-l border-border bg-card overflow-y-auto p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 style={{ fontSize: '14px' }} className="text-foreground">Counterfactual Simulator</h3>
            <button onClick={() => setSimOpen(false)} className="text-muted-foreground hover:text-foreground">✕</button>
          </div>
          <p className="text-muted-foreground mb-4" style={{ fontSize: '11px' }}>
            Adjust parameters to simulate alternative clinical trajectories. All outputs are AI-generated and require clinical interpretation.
          </p>

          <div className="space-y-4">
            <div>
              <label className="text-muted-foreground block mb-1" style={{ fontSize: '11px' }}>Dexamethasone Dose (mg BID)</label>
              <input type="range" min={0} max={100} value={simParam} onChange={e => setSimParam(Number(e.target.value))}
                className="w-full accent-primary" />
              <div className="flex justify-between text-muted-foreground font-mono" style={{ fontSize: '10px' }}>
                <span>0</span><span>{((simParam / 100) * 16).toFixed(0)} mg</span><span>16</span>
              </div>
            </div>

            <div>
              <label className="text-muted-foreground block mb-1" style={{ fontSize: '11px' }}>Bevacizumab Addition</label>
              <div className="flex gap-2">
                <button className="flex-1 px-3 py-1.5 rounded border border-border text-muted-foreground hover:border-primary hover:text-primary transition-colors" style={{ fontSize: '11px' }}>No</button>
                <button className="flex-1 px-3 py-1.5 rounded border border-primary bg-primary/15 text-primary" style={{ fontSize: '11px' }}>Yes</button>
              </div>
            </div>

            <div>
              <label className="text-muted-foreground block mb-1" style={{ fontSize: '11px' }}>Sleep Intervention</label>
              <select className="w-full bg-secondary border border-border rounded px-2 py-1.5 text-foreground" style={{ fontSize: '11px' }}>
                <option>None</option>
                <option>Melatonin 3mg QHS</option>
                <option>CBT-I Protocol</option>
                <option>Combined</option>
              </select>
            </div>

            <div>
              <label className="text-muted-foreground block mb-1" style={{ fontSize: '11px' }}>Simulation Horizon</label>
              <select className="w-full bg-secondary border border-border rounded px-2 py-1.5 text-foreground" style={{ fontSize: '11px' }}>
                <option>24 hours</option>
                <option>48 hours</option>
                <option>72 hours</option>
                <option>7 days</option>
              </select>
            </div>

            <div className="flex gap-2">
              <button onClick={runSim}
                className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                style={{ fontSize: '12px' }} disabled={simRunning}>
                {simRunning ? (
                  <><span className="w-3 h-3 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" /> Running...</>
                ) : (
                  <><Play size={12} /> Run Simulation</>
                )}
              </button>
              <button className="px-3 py-2 rounded border border-border text-muted-foreground hover:text-foreground transition-colors">
                <RotateCcw size={12} />
              </button>
            </div>

            {/* Simulated result */}
            {!simRunning && (
              <div className="border border-border rounded-lg p-3 bg-secondary/30">
                <div className="text-muted-foreground mb-2" style={{ fontSize: '10px', letterSpacing: '0.05em' }}>SIMULATED OUTCOME</div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground" style={{ fontSize: '11px' }}>Deterioration Δ</span>
                    <span className="font-mono text-[var(--risk-low)]" style={{ fontSize: '12px' }}>-12.4%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground" style={{ fontSize: '11px' }}>Edema Volume Δ</span>
                    <span className="font-mono text-[var(--risk-low)]" style={{ fontSize: '12px' }}>-8.2 cm³</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground" style={{ fontSize: '11px' }}>Cognitive Score Δ</span>
                    <span className="font-mono text-[var(--risk-moderate)]" style={{ fontSize: '12px' }}>+2.1 pts</span>
                  </div>
                  <div className="mt-2 pt-2 border-t border-border flex items-center gap-1">
                    <span className="px-1.5 py-0.5 rounded bg-[var(--risk-moderate-bg)] text-[var(--risk-moderate)] font-mono" style={{ fontSize: '9px' }}>CI 62%</span>
                    <span className="text-muted-foreground" style={{ fontSize: '10px' }}>Monte Carlo · 10k iterations</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
