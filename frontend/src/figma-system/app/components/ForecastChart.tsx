// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { TrendingUp } from 'lucide-react';
import type { AnalysisResult } from '../types/analysis';

interface ForecastChartProps {
  analysisResult?: AnalysisResult | null;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const p = payload[0]?.payload;
  return (
    <div className="rounded-xl border border-border bg-card/95 backdrop-blur-sm p-3 text-xs shadow-xl">
      <p className="font-mono font-medium text-foreground mb-2 pb-1 border-b border-border">{label}</p>
      <p style={{ color: 'var(--risk-high)' }}>Predicted: {(p.predicted * 100).toFixed(1)}%</p>
      <p className="text-muted-foreground">80% CI: {(p.lower * 100).toFixed(1)}% - {(p.upper * 100).toFixed(1)}%</p>
      {p.actual != null && <p style={{ color: 'var(--chart-2)' }}>Actual: {(p.actual * 100).toFixed(1)}%</p>}
    </div>
  );
};

export function ForecastChart({ analysisResult }: ForecastChartProps) {
  const forecastData = analysisResult
    ? analysisResult.trajectory.map((risk, index) => ({
        time: `${(index + 1) * 6}m`,
        predicted: risk,
        lower: analysisResult.confidence_bands?.lower?.[index] ?? Math.max(0, risk - 0.08),
        upper: analysisResult.confidence_bands?.upper?.[index] ?? Math.min(1, risk + 0.08),
      }))
    : [];

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">12-Month Deterioration Trajectory</h3>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <span style={{ background: 'var(--risk-high)', width: 10, height: 2, display: 'inline-block', borderRadius: 1 }} />
            Predicted
          </span>
          <span className="flex items-center gap-1">
            <span style={{ background: 'rgba(239,68,68,0.2)', width: 10, height: 8, display: 'inline-block', borderRadius: 1 }} />
            80% CI
          </span>
          <span className="flex items-center gap-1">
            <span style={{ background: 'var(--chart-2)', width: 10, height: 2, display: 'inline-block', borderRadius: 1 }} />
            Actual
          </span>
        </div>
      </div>
      {!analysisResult ? (
        <div className="flex h-[220px] flex-col items-center justify-center gap-3">
          <div className="w-12 h-12 rounded-full border-2 border-dashed border-border flex items-center justify-center">
            <TrendingUp size={20} className="text-muted-foreground" />
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-foreground">No trajectory data yet</div>
            <div className="text-xs text-muted-foreground mt-0.5">Enter patient data and run analysis</div>
          </div>
        </div>
      ) : (
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={forecastData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--risk-high)" stopOpacity={0.25} />
              <stop offset="100%" stopColor="var(--risk-high)" stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="time" tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0.75} stroke="var(--risk-critical)" strokeDasharray="4 4" strokeOpacity={0.5} label={{ value: 'Critical', fill: 'var(--risk-critical)', fontSize: 10 }} />
          <Area type="monotone" dataKey="upper" stroke="none" fill="url(#ciGrad)" isAnimationActive={true} animationDuration={800} animationEasing="ease-out" />
          <Area type="monotone" dataKey="lower" stroke="none" fill="var(--background)" isAnimationActive={true} animationDuration={800} animationEasing="ease-out" />
          <Area type="monotone" dataKey="predicted" stroke="var(--risk-high)" strokeWidth={2} fill="none" dot={false} isAnimationActive={true} animationDuration={800} animationEasing="ease-out" />
          <Area type="monotone" dataKey="actual" stroke="var(--chart-2)" strokeWidth={2} strokeDasharray="4 4" fill="none" dot={{ fill: 'var(--chart-2)', r: 3 }} isAnimationActive={true} animationDuration={800} animationEasing="ease-out" />
        </AreaChart>
      </ResponsiveContainer>
      )}
    </div>
  );
}
