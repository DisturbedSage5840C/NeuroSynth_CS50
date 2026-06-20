// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Area, AreaChart, CartesianGrid, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import './v3.css';

const MONTHS = [6, 12, 18, 24, 30, 36, 42, 48];

const DISEASE_TABS = [
  { key: 'current',    label: 'Current',     color: 'var(--accent-primary)' },
  { key: 'ad',         label: "Alzheimer's", color: 'var(--color-ad)' },
  { key: 'pd',         label: "Parkinson's", color: 'var(--color-pd)' },
  { key: 'ms',         label: 'MS',          color: 'var(--color-ms)' },
  { key: 'als',        label: 'ALS',         color: 'var(--color-als)' },
];

const REFERENCE_BANDS: Record<string, { lower: number; upper: number }> = {
  current: { lower: 0.15, upper: 0.45 },
  ad:      { lower: 0.35, upper: 0.75 },
  pd:      { lower: 0.25, upper: 0.65 },
  ms:      { lower: 0.20, upper: 0.55 },
  als:     { lower: 0.45, upper: 0.85 },
};

function projectTrajectory(baseValues: number[], disease: string): number[] {
  const multipliers: Record<string, number> = {
    ad: 1.15, pd: 1.05, ms: 1.08, als: 1.25,
  };
  const m = multipliers[disease] ?? 1.0;
  return baseValues.map((v, i) => Math.min(1, v * m * (1 + i * 0.015)));
}

// Intervention model: how each slider delta adjusts trajectory probability.
// Positive delta on a protective factor (activity, sleep) reduces risk.
// physicalActivity: 0–10, sleepQuality: 0–10, bmi: 15–40
const INTERVENTION_EFFECTS = {
  physicalActivity: -0.018,  // per unit increase → −1.8% risk per month-step
  sleepQuality:     -0.012,
  bmi:               0.008,  // higher BMI → increased risk
};

function applyInterventions(
  base: number[],
  deltas: { physicalActivity: number; sleepQuality: number; bmi: number },
): number[] {
  const adj =
    deltas.physicalActivity * INTERVENTION_EFFECTS.physicalActivity +
    deltas.sleepQuality     * INTERVENTION_EFFECTS.sleepQuality +
    deltas.bmi              * INTERVENTION_EFFECTS.bmi;
  return base.map((v) => Math.min(1, Math.max(0, v + adj)));
}

interface Props {
  months?: number[];
  values?: number[];
  bandsLower?: number[];
  bandsUpper?: number[];
}

export function TrajectoryChartV3({
  months = MONTHS,
  values = [],
  bandsLower = [],
  bandsUpper = [],
}: Props) {
  const [activeTab, setActiveTab]             = useState<string>('current');
  const [showIntervention, setShowIntervention] = useState(false);
  const [activity, setActivity]               = useState(5);
  const [sleep, setSleep]                     = useState(5);
  const [bmi, setBmi]                         = useState(25);

  const hasData = values.length > 0;

  const baseValues = useMemo(() => {
    if (!hasData) return [];
    return activeTab === 'current' ? values : projectTrajectory(values, activeTab);
  }, [values, activeTab, hasData]);

  const interventionValues = useMemo(() => {
    if (!hasData || !showIntervention) return null;
    // Compute deltas from neutral (activity=5, sleep=5, bmi=25)
    return applyInterventions(baseValues, {
      physicalActivity: activity - 5,
      sleepQuality: sleep - 5,
      bmi: bmi - 25,
    });
  }, [baseValues, showIntervention, activity, sleep, bmi, hasData]);

  const tabColor = DISEASE_TABS.find((t) => t.key === activeTab)?.color ?? 'var(--accent-primary)';
  const ref = REFERENCE_BANDS[activeTab] ?? REFERENCE_BANDS.current;

  const chartData = months.map((m, i) => ({
    month: `Mo ${m}`,
    risk:  baseValues[i] ?? null,
    lower: bandsLower[i] ?? (baseValues[i] != null ? Math.max(0, baseValues[i] - 0.07) : null),
    upper: bandsUpper[i] ?? (baseValues[i] != null ? Math.min(1, baseValues[i] + 0.07) : null),
    refLower: ref.lower,
    refUpper: ref.upper,
    intervention: interventionValues ? (interventionValues[i] ?? null) : null,
  }));

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">48-Month Risk Trajectory</h3>
        <div className="flex items-center gap-3">
          {hasData && (
            <button
              type="button"
              onClick={() => setShowIntervention((s) => !s)}
              className={`text-xs px-2.5 py-1 rounded-md border transition-colors ${
                showIntervention
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:border-primary/40'
              }`}
            >
              Intervention
            </button>
          )}
          {hasData && (
            <span className="text-xs font-mono text-muted-foreground">
              {Math.round((baseValues[baseValues.length - 1] ?? 0) * 100)}% at 48 mo
            </span>
          )}
        </div>
      </div>

      {/* Disease tabs */}
      <div className="traj-tabs">
        {DISEASE_TABS.map((tab) => (
          <motion.button
            key={tab.key}
            type="button"
            className={`traj-tab${activeTab === tab.key ? ' traj-tab-active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
            whileTap={{ scale: 0.95 }}
          >
            {tab.label}
          </motion.button>
        ))}
      </div>

      {/* Chart */}
      <div className="traj-chart-area">
        {!hasData ? (
          <div className="flex items-center justify-center h-full text-xs text-muted-foreground">
            Run analysis to see trajectory
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <defs>
                <linearGradient id="traj-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={tabColor} stopOpacity={0.25} />
                  <stop offset="95%" stopColor={tabColor} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="traj-int" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="var(--risk-low)" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="var(--risk-low)" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} />
              <XAxis dataKey="month" tick={{ fontSize: 9, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                tick={{ fontSize: 9, fill: 'var(--text-tertiary)' }}
                axisLine={false} tickLine={false} width={36}
              />
              <Tooltip
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11 }}
                formatter={(v: number, name: string) => {
                  const labels: Record<string, string> = {
                    risk: 'Risk', lower: 'CI Lower', upper: 'CI Upper',
                    refLower: 'Pop. Lower', refUpper: 'Pop. Upper',
                    intervention: 'Intervention',
                  };
                  return [`${Math.round(v * 100)}%`, labels[name] ?? name];
                }}
              />
              <Area type="monotone" dataKey="refUpper" stroke="none" fill="rgba(100,116,139,0.08)" fillOpacity={1} isAnimationActive={false} />
              <Area type="monotone" dataKey="refLower" stroke="none" fill="var(--bg-base)" fillOpacity={1} isAnimationActive={false} />
              <Area type="monotone" dataKey="upper" stroke="none" fill="url(#traj-fill)" fillOpacity={1} />
              <Line type="monotone" dataKey="risk" stroke={tabColor} strokeWidth={2} dot={false} activeDot={{ r: 4, fill: tabColor }} />
              {interventionValues && (
                <Line
                  type="monotone"
                  dataKey="intervention"
                  stroke="var(--risk-low)"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  dot={false}
                  activeDot={{ r: 4, fill: 'var(--risk-low)' }}
                />
              )}
              <ReferenceLine y={0.6} stroke="var(--risk-high)" strokeDasharray="4 3" strokeOpacity={0.4}
                label={{ value: '60%', fill: 'var(--risk-high)', fontSize: 9, position: 'insideRight' }} />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Intervention sliders */}
      {showIntervention && hasData && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="traj-intervention"
        >
          <div className="traj-intervention-title">Intervention Scenario</div>
          {[
            { label: 'Physical Activity', val: activity, set: setActivity, min: 0, max: 10 },
            { label: 'Sleep Quality',     val: sleep,    set: setSleep,    min: 0, max: 10 },
            { label: 'BMI',               val: bmi,      set: setBmi,      min: 15, max: 40 },
          ].map(({ label, val, set, min, max }) => (
            <div key={label} className="traj-slider-row">
              <span className="traj-slider-label">{label}</span>
              <input
                type="range"
                aria-label={label}
                className="traj-slider"
                min={min}
                max={max}
                value={val}
                onChange={(e) => set(Number(e.target.value))}
              />
              <span className="traj-slider-val">{val}</span>
            </div>
          ))}
          <p className="text-xs text-muted-foreground mt-2">
            Dashed line shows projected risk under this scenario. Effects are approximate.
          </p>
        </motion.div>
      )}

      {activeTab !== 'current' && hasData && (
        <p className="mt-2 text-xs text-muted-foreground">
          Projection assumes <strong style={{ color: tabColor }}>
            {DISEASE_TABS.find((t) => t.key === activeTab)?.label}
          </strong> disease trajectory model.
        </p>
      )}
    </div>
  );
}
