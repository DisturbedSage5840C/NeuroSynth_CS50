// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useQuery } from '@tanstack/react-query';
import './v3.css';

const API = import.meta.env.VITE_API_BASE_URL ?? '';

interface DiseasePrevalence { name: string; value: number; count: number; color: string; }
interface AgeGroup { range: string; ad: number; pd: number; ms: number; ep: number; als: number; hd: number; }

interface CohortStatsData {
  total_patients: number;
  data_sources: number;
  prevalence: DiseasePrevalence[];
  age_distribution: AgeGroup[];
  feature_count: number;
}

const FALLBACK: CohortStatsData = {
  total_patients: 16026,
  data_sources: 11,
  feature_count: 56,
  prevalence: [
    { name: "Alzheimer's Disease", value: 38, count: 1013, color: '#818cf8' },
    { name: "Parkinson's Disease", value: 28, count: 748,  color: '#34d399' },
    { name: 'Multiple Sclerosis',  value: 14, count: 374,  color: '#fb923c' },
    { name: 'Epilepsy',            value: 12, count: 320,  color: '#a78bfa' },
    { name: 'ALS',                 value: 5,  count: 133,  color: '#f87171' },
    { name: "Huntington's Disease",value: 3,  count: 80,   color: '#fbbf24' },
  ],
  age_distribution: [
    { range: '20–30', ad: 2,  pd: 3,  ms: 8,  ep: 12, als: 1, hd: 1 },
    { range: '30–40', ad: 4,  pd: 6,  ms: 18, ep: 14, als: 2, hd: 1 },
    { range: '40–50', ad: 8,  pd: 14, ms: 22, ep: 11, als: 4, hd: 2 },
    { range: '50–60', ad: 18, pd: 24, ms: 19, ep: 9,  als: 8, hd: 3 },
    { range: '60–70', ad: 32, pd: 28, ms: 14, ep: 7,  als: 12,hd: 5 },
    { range: '70+',   ad: 36, pd: 25, ms: 9,  ep: 5,  als: 8, hd: 3 },
  ],
};

export function CohortStats() {
  const { data } = useQuery<CohortStatsData>({
    queryKey: ['cohort-stats'],
    queryFn: async () => {
      const res = await fetch(`${API}/v3/data/cohort/stats`);
      if (!res.ok) throw new Error('fetch failed');
      return res.json();
    },
    placeholderData: FALLBACK,
    staleTime: 5 * 60 * 1000,
  });

  const stats = data ?? FALLBACK;
  const maxPrevalence = Math.max(...stats.prevalence.map((p) => p.value));

  return (
    <div className="glass-card">
      {/* KPI strip */}
      <div className="cohort-kpi-row">
        {[
          { label: 'Patients', value: stats.total_patients.toLocaleString() },
          { label: 'Sources',  value: stats.data_sources },
          { label: 'Features', value: stats.feature_count },
        ].map(({ label, value }) => (
          <div key={label} className="data-badge">
            <span className="data-badge-value">{value}</span>
            <span className="data-badge-unit">{label}</span>
          </div>
        ))}
      </div>

      {/* Disease prevalence bars */}
      <div className="cohort-section-label">Disease Prevalence</div>
      <div className="cohort-prevalence-list">
        {stats.prevalence.map((d) => (
          <div key={d.name} className="cohort-prev-row">
            <span className="cohort-prev-name">{d.name}</span>
            <div className="progress-track" style={{ flex: 1 }}>
              <div
                className="progress-fill"
                style={{
                  width: `${(d.value / maxPrevalence) * 100}%`,
                  background: d.color,
                }}
              />
            </div>
            <span className="cohort-prev-pct" style={{ color: d.color }}>{d.value}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
