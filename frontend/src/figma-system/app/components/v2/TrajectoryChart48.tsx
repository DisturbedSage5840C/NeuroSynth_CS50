// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { motion } from 'framer-motion';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface TrajectoryChart48Props {
  months?: number[];
  values?: number[];
  bandsLower?: number[];
  bandsUpper?: number[];
}

export function TrajectoryChart48({
  months = [6, 12, 18, 24, 30, 36, 42, 48],
  values = [],
  bandsLower = [],
  bandsUpper = [],
}: TrajectoryChart48Props) {
  const data = months.map((m, i) => ({
    month: `Mo ${m}`,
    risk: values[i] ?? null,
    lower: bandsLower[i] ?? (values[i] ? Math.max(0, values[i] - 0.08) : null),
    upper: bandsUpper[i] ?? (values[i] ? Math.min(1, values[i] + 0.08) : null),
  }));

  const hasData = values.length > 0;

  const riskColor = (v: number) => {
    if (v >= 0.7) return '#ef4444';
    if (v >= 0.5) return '#f59e0b';
    return '#22c55e';
  };

  const latestRisk = values[values.length - 1] ?? 0;

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-medium text-foreground">48-Month Risk Trajectory</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Predicted progression with 95% confidence bands
          </p>
        </div>
        {hasData && (
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="flex items-center gap-2"
          >
            <div
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: riskColor(latestRisk) }}
            />
            <span className="text-sm font-mono font-semibold" style={{ color: riskColor(latestRisk) }}>
              {Math.round(latestRisk * 100)}% at 48mo
            </span>
          </motion.div>
        )}
      </div>

      {hasData ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="riskGradient48" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#818cf8" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="bandGradient48" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#818cf8" stopOpacity={0.08} />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                vertical={false}
              />
              <XAxis
                dataKey="month"
                tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--card-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(value: number, name: string) => [
                  `${(value * 100).toFixed(1)}%`,
                  name === 'risk' ? 'Risk' : name === 'upper' ? 'Upper CI' : 'Lower CI',
                ]}
              />
              {/* Confidence band */}
              <Area
                type="monotone"
                dataKey="upper"
                stroke="none"
                fill="url(#bandGradient48)"
                fillOpacity={1}
              />
              <Area
                type="monotone"
                dataKey="lower"
                stroke="none"
                fill="var(--card)"
                fillOpacity={1}
              />
              {/* Main risk line */}
              <Area
                type="monotone"
                dataKey="risk"
                stroke="#818cf8"
                strokeWidth={2.5}
                fill="url(#riskGradient48)"
                fillOpacity={1}
                dot={{ r: 4, fill: '#818cf8', stroke: 'var(--card)', strokeWidth: 2 }}
                activeDot={{ r: 6, fill: '#818cf8', stroke: '#fff', strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>
      ) : (
        <div className="flex flex-col items-center justify-center h-[240px] text-muted-foreground">
          <div className="text-xs">Run analysis to generate trajectory forecast</div>
        </div>
      )}

      {/* Legend */}
      {hasData && (
        <div className="flex items-center justify-center gap-6 mt-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-0.5 rounded-full" style={{ backgroundColor: '#818cf8' }} />
            Predicted risk
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-3 rounded-sm" style={{ backgroundColor: '#818cf8', opacity: 0.1 }} />
            95% confidence
          </div>
        </div>
      )}
    </div>
  );
}
