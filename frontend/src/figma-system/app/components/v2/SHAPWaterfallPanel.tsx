// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { motion } from 'framer-motion';
import { FEATURE_MAP, featureLabel } from '@/lib/featureSchema';
import { FeatureLegend } from './FeatureLegend';

interface SHAPValue {
  feature: string;
  value: number;
}

interface SHAPWaterfallPanelProps {
  shapValues: SHAPValue[];
  maxDisplay?: number;
}

export function SHAPWaterfallPanel({ shapValues, maxDisplay = 10 }: SHAPWaterfallPanelProps) {
  const sorted = [...shapValues]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, maxDisplay);

  const maxAbs = Math.max(...sorted.map((s) => Math.abs(s.value)), 0.01);

  // Encoded (categorical/boolean) features in view need a legend so 0/1/2 are clear.
  const encodedFeatures = sorted
    .map((s) => s.feature)
    .filter((key) => FEATURE_MAP[key]?.values);

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-foreground">SHAP Feature Attribution</h3>
        <span className="text-xs text-muted-foreground font-mono">
          {sorted.length} features
        </span>
      </div>

      <div className="space-y-2">
        {sorted.map((item, i) => {
          const isPositive = item.value > 0;
          const barWidth = (Math.abs(item.value) / maxAbs) * 100;

          return (
            <motion.div
              key={item.feature}
              initial={{ opacity: 0, x: isPositive ? 20 : -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.06, duration: 0.4 }}
              className="flex items-center gap-3"
            >
              {/* Feature name (human-readable) */}
              {(() => {
                const label = featureLabel(item.feature);
                return (
                  <div
                    className="text-xs font-mono text-muted-foreground text-right shrink-0"
                    style={{ width: 160 }}
                    title={`${label} (${item.feature})`}
                  >
                    {label.length > 22 ? label.slice(0, 20) + '…' : label}
                  </div>
                );
              })()}

              {/* Bar container (centered) */}
              <div className="flex-1 flex items-center h-6 relative">
                {/* Center line */}
                <div
                  className="absolute left-1/2 top-0 bottom-0 w-px"
                  style={{ backgroundColor: 'var(--border-strong)' }}
                />

                {/* Bar */}
                <div className="absolute left-1/2 h-5 flex items-center" style={{
                  transform: isPositive ? 'none' : 'translateX(-100%)',
                }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth * 0.5}%` }}
                    transition={{ delay: i * 0.06 + 0.2, duration: 0.5, ease: 'easeOut' }}
                    className="h-full rounded-sm"
                    style={{
                      minWidth: 4,
                      maxWidth: '50%',
                      width: `${barWidth * 0.5}%`,
                      backgroundColor: isPositive ? '#ef4444' : '#22c55e',
                      opacity: 0.8,
                    }}
                  />
                </div>
              </div>

              {/* Value */}
              <div
                className="text-xs font-mono shrink-0 w-16 text-right"
                style={{ color: isPositive ? '#ef4444' : '#22c55e' }}
              >
                {isPositive ? '+' : ''}{item.value.toFixed(4)}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 pt-3 border-t border-border">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#ef4444', opacity: 0.8 }} />
          Increases risk
        </div>
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#22c55e', opacity: 0.8 }} />
          Decreases risk
        </div>
      </div>

      {/* Encoding reference for any categorical/boolean features shown above */}
      <FeatureLegend visibleFeatures={encodedFeatures} />
    </div>
  );
}
