// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { motion } from 'framer-motion';
import { ArrowDown, Lightbulb } from 'lucide-react';

interface Counterfactual {
  feature: string;
  current_value: number;
  target_value: number;
  risk_delta: number;
  interpretation?: string;
}

interface CounterfactualPanelProps {
  counterfactuals: Counterfactual[];
}

export function CounterfactualPanel({ counterfactuals }: CounterfactualPanelProps) {
  if (!counterfactuals.length) {
    return (
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="text-sm font-medium text-foreground mb-3">Counterfactual Recommendations</h3>
        <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
          <Lightbulb size={24} className="mb-2 opacity-40" />
          <span className="text-xs">Run analysis to generate recommendations</span>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Lightbulb size={16} className="text-amber-400" />
          <h3 className="text-sm font-medium text-foreground">What-If Recommendations</h3>
        </div>
        <span className="text-xs text-muted-foreground font-mono">
          {counterfactuals.length} interventions
        </span>
      </div>

      <div className="space-y-3">
        {counterfactuals.map((cf, i) => (
          <motion.div
            key={`${cf.feature}-${i}`}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="rounded-lg border border-border p-4"
            style={{ background: 'var(--card-elevated)' }}
          >
            <div className="flex items-start justify-between mb-2">
              <div>
                <div className="text-sm font-medium text-foreground">{cf.feature}</div>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: 'var(--secondary)' }}>
                    {cf.current_value.toFixed(2)}
                  </span>
                  <span className="text-muted-foreground text-xs">→</span>
                  <span
                    className="text-xs font-mono px-2 py-0.5 rounded"
                    style={{ background: 'rgba(34, 197, 94, 0.15)', color: '#22c55e' }}
                  >
                    {cf.target_value.toFixed(2)}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-1 px-2 py-1 rounded-md" style={{ background: 'rgba(34, 197, 94, 0.12)' }}>
                <ArrowDown size={12} className="text-green-400" />
                <span className="text-xs font-mono font-semibold text-green-400">
                  {Math.abs(cf.risk_delta * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {cf.interpretation && (
              <div className="text-xs text-muted-foreground mt-2 leading-relaxed">
                {cf.interpretation}
              </div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
