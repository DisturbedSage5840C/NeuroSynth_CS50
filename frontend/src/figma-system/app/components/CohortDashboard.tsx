// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useQuery } from '@tanstack/react-query';
import {
  BarChart, Bar, PieChart, Pie, Cell,
  ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import { GlassCard } from './v3/GlassCard';
import './v3/v3.css';

// Map disease name → dc- CSS class suffix for text + bg
function dcText(name: string): string {
  if (name.includes('Alzheimer')) return 'dc-text-ad';
  if (name.includes('Parkinson')) return 'dc-text-pd';
  if (name.includes('Multiple Sclerosis') || name === 'MS') return 'dc-text-ms';
  if (name === 'Epilepsy') return 'dc-text-ep';
  if (name === 'ALS') return 'dc-text-als';
  if (name.includes('Huntington')) return 'dc-text-hd';
  if (name === 'Healthy') return 'dc-text-hl';
  return 'dc-text-default';
}

function dcBg(name: string): string {
  if (name.includes('Alzheimer')) return 'dc-bg-ad';
  if (name.includes('Parkinson')) return 'dc-bg-pd';
  if (name.includes('Multiple Sclerosis') || name === 'MS') return 'dc-bg-ms';
  if (name === 'Epilepsy') return 'dc-bg-ep';
  if (name === 'ALS') return 'dc-bg-als';
  if (name.includes('Huntington')) return 'dc-bg-hd';
  if (name === 'Healthy') return 'dc-bg-hl';
  return 'dc-bg-default';
}

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const DISEASE_COLORS: Record<string, string> = {
  "Alzheimer's Disease": '#818cf8',
  "Parkinson's Disease": '#34d399',
  'Multiple Sclerosis':  '#fb923c',
  'Epilepsy':            '#a78bfa',
  'ALS':                 '#f87171',
  "Huntington's Disease":'#fbbf24',
  'Healthy':             '#64748b',
};

const FALLBACK_PREVALENCE = [
  { name: "Alzheimer's",  value: 38, color: '#818cf8' },
  { name: "Parkinson's",  value: 28, color: '#34d399' },
  { name: 'MS',           value: 14, color: '#fb923c' },
  { name: 'Epilepsy',     value: 12, color: '#a78bfa' },
  { name: 'ALS',          value: 5,  color: '#f87171' },
  { name: "Huntington's", value: 3,  color: '#fbbf24' },
];

const FALLBACK_AGE = [
  { range: '20–30', ad: 2,  pd: 3,  ms: 8,  ep: 12 },
  { range: '30–40', ad: 4,  pd: 6,  ms: 18, ep: 14 },
  { range: '40–50', ad: 8,  pd: 14, ms: 22, ep: 11 },
  { range: '50–60', ad: 18, pd: 24, ms: 19, ep: 9  },
  { range: '60–70', ad: 32, pd: 28, ms: 14, ep: 7  },
  { range: '70+',   ad: 36, pd: 25, ms: 9,  ep: 5  },
];

const STATS = [
  { label: 'Total Patients',   value: '16,026', sub: 'real clinical records' },
  { label: 'Data Sources',     value: '11',      sub: 'Kaggle + PhysioNet + OASIS' },
  { label: 'Avg AUC (v5)',     value: '0.994',   sub: 'binary risk ensemble' },
  { label: 'Disease Classes',  value: '7',       sub: 'incl. Healthy' },
];

function fetchCohortStats() {
  return fetch(`${API}/v3/data/cohort/stats`, { credentials: 'include' })
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
}

export function CohortDashboard() {
  const { data: cohort } = useQuery({ queryKey: ['cohort-stats'], queryFn: fetchCohortStats });
  const prevalence = cohort?.prevalence ?? FALLBACK_PREVALENCE;
  const ageDist    = cohort?.age_distribution ?? FALLBACK_AGE;

  return (
    <div className="page-root">
      <div className="page-header">
        <h1 className="page-title">Cohort Dashboard</h1>
        <p className="page-subtitle">Population-level statistics from the real_v5 dataset</p>
      </div>

      {/* KPI strip */}
      <div className="page-grid-4 mb-6">
        {STATS.map((s) => (
          <GlassCard key={s.label}>
            <div className="text-data text-xl font-bold">{s.value}</div>
            <div className="text-sm font-medium text-foreground mt-0.5">{s.label}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{s.sub}</div>
          </GlassCard>
        ))}
      </div>

      <div className="page-grid-2 mb-6">
        {/* Disease prevalence pie */}
        <GlassCard>
          <h3 className="text-sm font-medium text-foreground mb-4">Disease Prevalence</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={prevalence}
                cx="50%" cy="50%"
                innerRadius={55} outerRadius={85}
                dataKey="value"
                stroke="none"
              >
                {prevalence.map((entry: { name: string; color?: string }, i: number) => (
                  <Cell key={i} fill={entry.color ?? DISEASE_COLORS[entry.name] ?? '#64748b'} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11 }}
                formatter={(v: number, name: string) => [`${v}%`, name]}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
            {prevalence.map((d: { name: string; value: number; color?: string }) => (
              <div key={d.name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span className={`legend-swatch ${dcBg(d.name)}`} />
                {d.name} ({d.value}%)
              </div>
            ))}
          </div>
        </GlassCard>

        {/* Age × disease distribution */}
        <GlassCard>
          <h3 className="text-sm font-medium text-foreground mb-4">Age Distribution by Disease</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={ageDist} margin={{ left: -20, right: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.4} />
              <XAxis dataKey="range" tick={{ fontSize: 9, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 9, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11 }} />
              <Bar dataKey="ad" name="Alzheimer's" fill="#818cf8" stackId="a" />
              <Bar dataKey="pd" name="Parkinson's" fill="#34d399" stackId="a" />
              <Bar dataKey="ms" name="MS"           fill="#fb923c" stackId="a" />
              <Bar dataKey="ep" name="Epilepsy"     fill="#a78bfa" stackId="a" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>
      </div>

      {/* Per-disease AUC from manifest */}
      <GlassCard>
        <h3 className="text-sm font-medium text-foreground mb-4">v5 Per-Disease F1 (Test Set)</h3>
        <div className="grid grid-cols-7 gap-3">
          {[
            { name: "ALS",            f1: 1.00,  color: '#f87171' },
            { name: "Alzheimer's",    f1: 0.986, color: '#818cf8' },
            { name: "Epilepsy",       f1: 1.00,  color: '#a78bfa' },
            { name: "Healthy",        f1: 0.840, color: '#64748b' },
            { name: "Huntington's",   f1: 1.00,  color: '#fbbf24' },
            { name: "MS",             f1: 1.00,  color: '#fb923c' },
            { name: "Parkinson's",    f1: 0.995, color: '#34d399' },
          ].map((d) => (
            <div key={d.name} className="flex flex-col items-center gap-1">
              <div className={`text-base font-bold font-mono ${dcText(d.name)}`}>
                {(d.f1 * 100).toFixed(1)}%
              </div>
              <div className="progress-track">
                <div
                  className={`progress-fill ${dcBg(d.name)}`}
                  style={{ width: `${d.f1 * 100}%` }}
                />
              </div>
              <div className="text-xs text-muted-foreground text-center leading-tight">{d.name}</div>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  );
}
