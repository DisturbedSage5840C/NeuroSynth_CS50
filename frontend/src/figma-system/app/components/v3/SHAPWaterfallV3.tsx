// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useState } from 'react';
import { motion } from 'framer-motion';
import { FEATURE_MAP, featureLabel } from '@/lib/featureSchema';
import '../v3/v3.css';

interface SHAPValue { feature: string; value: number; }

interface Props {
  shapValues: SHAPValue[];
  maxDisplay?: number;
  causalFeatures?: string[];  // features present in causal graph
}

// Clinical significance based on feature importance
function significance(abs: number, maxAbs: number): 'high' | 'medium' | 'low' {
  const ratio = abs / maxAbs;
  if (ratio >= 0.5) return 'high';
  if (ratio >= 0.2) return 'medium';
  return 'low';
}

const SIG_LABELS = { high: 'High', medium: 'Med', low: 'Low' };
const SIG_CLASSES = { high: 'shap-sig-high', medium: 'shap-sig-medium', low: 'shap-sig-low' };

export function SHAPWaterfallV3({ shapValues, maxDisplay = 10, causalFeatures = [] }: Props) {
  const [showCausalOnly, setShowCausalOnly] = useState(false);

  const sorted = [...shapValues]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, maxDisplay);

  const displayed = showCausalOnly
    ? sorted.filter((s) => causalFeatures.includes(s.feature))
    : sorted;

  const maxAbs = Math.max(...sorted.map((s) => Math.abs(s.value)), 0.01);
  const hasCausal = causalFeatures.length > 0;

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-foreground">SHAP Feature Attribution</h3>
        <div className="flex items-center gap-3">
          {hasCausal && (
            <label className="flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                className="sr-only"
                checked={showCausalOnly}
                onChange={(e) => setShowCausalOnly(e.target.checked)}
              />
              <div
                className="w-8 h-4 rounded-full relative transition-colors"
                style={{ background: showCausalOnly ? 'var(--accent-primary)' : 'var(--bg-subtle)' }}
              >
                <div
                  className="absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform"
                  style={{ transform: showCausalOnly ? 'translateX(17px)' : 'translateX(2px)' }}
                />
              </div>
              <span className="text-xs text-muted-foreground">Causal only</span>
            </label>
          )}
          <span className="text-xs text-muted-foreground font-mono">{displayed.length} features</span>
        </div>
      </div>

      <div className="space-y-2">
        {displayed.map((item, i) => {
          const isPos = item.value > 0;
          const barPct = (Math.abs(item.value) / maxAbs) * 100;
          const color = isPos ? '#ef4444' : '#10b981';
          const sig = significance(Math.abs(item.value), maxAbs);
          const label = featureLabel(item.feature);
          const isCausal = causalFeatures.includes(item.feature);

          return (
            <motion.div
              key={item.feature}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
              className="flex flex-col gap-1"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 min-w-0">
                  {isCausal && <span className="shap-causal-dot flex-shrink-0" title="Causal driver" />}
                  <span className="text-xs text-foreground truncate" title={label}>{label}</span>
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className={`shap-significance-badge ${SIG_CLASSES[sig]}`}>
                    {SIG_LABELS[sig]}
                  </span>
                  <span className="text-xs font-mono text-muted-foreground w-14 text-right">
                    {item.value > 0 ? '+' : ''}{item.value.toFixed(4)}
                  </span>
                </div>
              </div>

              <div className="relative h-1.5 rounded-full overflow-hidden bg-secondary">
                <motion.div
                  className="absolute top-0 h-full rounded-full"
                  style={{
                    left: isPos ? '50%' : `${50 - barPct / 2}%`,
                    backgroundColor: color,
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${barPct / 2}%` }}
                  transition={{ duration: 0.6, delay: i * 0.04, ease: 'easeOut' }}
                />
              </div>
            </motion.div>
          );
        })}
      </div>

      {displayed.length === 0 && (
        <p className="text-xs text-muted-foreground text-center py-4">
          No causal features found in top SHAP values.
        </p>
      )}
    </div>
  );
}
