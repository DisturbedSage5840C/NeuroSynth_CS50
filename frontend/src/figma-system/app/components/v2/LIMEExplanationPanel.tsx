// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface LIMEWeight {
  feature: string;
  weight: number;
  direction: string;
}

interface LIMEExplanationPanelProps {
  limeValues: LIMEWeight[];
  maxDisplay?: number;
}

export function LIMEExplanationPanel({ limeValues, maxDisplay = 8 }: LIMEExplanationPanelProps) {
  const sorted = [...limeValues]
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, maxDisplay);

  const maxAbs = Math.max(...sorted.map((s) => Math.abs(s.weight)), 0.001);

  if (!sorted.length) {
    return (
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="text-sm font-medium text-foreground mb-3">LIME Local Explanation</h3>
        <div className="flex flex-col items-center justify-center py-8 text-muted-foreground text-xs">
          Run analysis to generate local explanations
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-medium text-foreground">LIME Local Explanation</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Perturbation-based feature importance for this patient
          </p>
        </div>
        <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}>
          {sorted.length} features
        </span>
      </div>

      <div className="space-y-2.5">
        {sorted.map((item, i) => {
          const isRisk = item.direction === 'increases_risk';
          const barWidth = (Math.abs(item.weight) / maxAbs) * 100;
          const color = isRisk ? '#ef4444' : '#22c55e';

          return (
            <motion.div
              key={item.feature}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="group"
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5">
                  {isRisk ? (
                    <TrendingUp size={12} style={{ color }} />
                  ) : (
                    <TrendingDown size={12} style={{ color }} />
                  )}
                  <span className="text-xs text-foreground font-medium">
                    {item.feature}
                  </span>
                </div>
                <span className="text-xs font-mono" style={{ color }}>
                  {item.weight > 0 ? '+' : ''}{item.weight.toFixed(4)}
                </span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--secondary)' }}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${barWidth}%` }}
                  transition={{ delay: i * 0.05 + 0.2, duration: 0.6, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: color, opacity: 0.7 }}
                />
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
