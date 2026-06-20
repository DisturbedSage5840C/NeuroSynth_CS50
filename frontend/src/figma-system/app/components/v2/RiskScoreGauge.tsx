// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { motion } from 'framer-motion';
import { useMemo } from 'react';

interface RiskScoreGaugeProps {
  probability: number;
  riskLevel: string;
  confidence: string;
  modelCount?: number;
  compact?: boolean;
}

const RISK_COLORS: Record<string, string> = {
  critical: '#ef4444',
  high: '#f97316',
  moderate: '#eab308',
  low: '#22c55e',
};

export function RiskScoreGauge({
  probability,
  riskLevel,
  confidence,
  modelCount = 5,
  compact = false,
}: RiskScoreGaugeProps) {
  const pct = Math.round(probability * 100);
  const level = riskLevel.toLowerCase();
  const color = RISK_COLORS[level] || RISK_COLORS.moderate;

  const { circumference, dashOffset, radius, size, strokeWidth } = useMemo(() => {
    const s = compact ? 120 : 180;
    const sw = compact ? 8 : 12;
    const r = (s - sw) / 2;
    const circ = 2 * Math.PI * r;
    const offset = circ * (1 - probability);
    return { circumference: circ, dashOffset: offset, radius: r, size: s, strokeWidth: sw };
  }, [probability, compact]);

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Circular gauge */}
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          {/* Background track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="var(--border)"
            strokeWidth={strokeWidth}
          />
          {/* Animated progress arc */}
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: dashOffset }}
            transition={{ duration: 1.2, ease: 'easeOut' }}
            style={{
              filter: `drop-shadow(0 0 8px ${color}40)`,
            }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="font-mono font-bold leading-none"
            style={{ fontSize: compact ? 28 : 44, color }}
          >
            {pct}%
          </motion.div>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="text-xs font-mono tracking-widest mt-1"
            style={{ color }}
          >
            {riskLevel.toUpperCase()}
          </motion.div>
        </div>
      </div>

      {/* Sub-metrics */}
      {!compact && (
        <motion.div
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="flex items-center gap-4 text-xs"
        >
          <div className="flex items-center gap-1.5">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: confidence === 'High' ? '#22c55e' : confidence === 'Medium' ? '#eab308' : '#ef4444' }}
            />
            <span className="text-muted-foreground">{confidence} confidence</span>
          </div>
          <div className="text-muted-foreground">
            <span className="font-mono">{modelCount}</span> models
          </div>
        </motion.div>
      )}
    </div>
  );
}
