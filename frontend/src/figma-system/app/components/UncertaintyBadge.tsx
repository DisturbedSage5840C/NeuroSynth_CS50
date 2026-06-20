// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { Info } from 'lucide-react';

interface UncertaintyBadgeProps {
  confidence: number;
  showLabel?: boolean;
  size?: 'sm' | 'md';
}

export function UncertaintyBadge({ confidence, showLabel = true, size = 'sm' }: UncertaintyBadgeProps) {
  const pct = Math.round(confidence * 100);
  const level = confidence >= 0.8 ? 'high' : confidence >= 0.6 ? 'medium' : 'low';
  const colors = {
    high: 'bg-[var(--risk-low-bg)] text-[var(--risk-low)] border-[var(--risk-low)]/20',
    medium: 'bg-[var(--risk-moderate-bg)] text-[var(--risk-moderate)] border-[var(--risk-moderate)]/20',
    low: 'bg-[var(--risk-critical-bg)] text-[var(--risk-critical)] border-[var(--risk-critical)]/20',
  };
  const iconSize = size === 'sm' ? 10 : 12;
  const padding = size === 'sm' ? 'px-1.5 py-0.5' : 'px-2 py-1';

  return (
    <span className={`inline-flex items-center gap-1 rounded border font-mono ${padding} ${colors[level]}`}
      style={{ fontSize: size === 'sm' ? '10px' : '11px' }}
      title={`AI confidence: ${pct}% (${level})`}>
      <Info size={iconSize} />
      {showLabel && <span>CI {pct}%</span>}
    </span>
  );
}

interface RiskBadgeProps {
  level: 'critical' | 'high' | 'moderate' | 'low';
  value?: string;
}

export function RiskBadge({ level, value }: RiskBadgeProps) {
  const config = {
    critical: { bg: 'bg-[var(--risk-critical-bg)]', text: 'text-[var(--risk-critical)]', border: 'border-[var(--risk-critical)]/20', label: 'CRITICAL' },
    high: { bg: 'bg-[var(--risk-high-bg)]', text: 'text-[var(--risk-high)]', border: 'border-[var(--risk-high)]/20', label: 'HIGH' },
    moderate: { bg: 'bg-[var(--risk-moderate-bg)]', text: 'text-[var(--risk-moderate)]', border: 'border-[var(--risk-moderate)]/20', label: 'MODERATE' },
    low: { bg: 'bg-[var(--risk-low-bg)]', text: 'text-[var(--risk-low)]', border: 'border-[var(--risk-low)]/20', label: 'LOW' },
  };
  const c = config[level];

  return (
    <span className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono ${c.bg} ${c.text} ${c.border}`}
      style={{ fontSize: '10px', letterSpacing: '0.05em' }}>
      <span className={`w-1.5 h-1.5 rounded-full ${level === 'critical' ? 'animate-pulse' : ''}`}
        style={{ backgroundColor: `var(--risk-${level})` }} />
      {value || c.label}
    </span>
  );
}
