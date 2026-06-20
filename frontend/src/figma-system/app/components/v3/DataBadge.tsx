// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import './v3.css';

interface DataBadgeProps {
  value: string | number;
  unit?: string;
  label?: string;
}

export function DataBadge({ value, unit, label }: DataBadgeProps) {
  return (
    <span className="data-badge" title={label}>
      <span className="data-badge-value">{value}</span>
      {unit && <span className="data-badge-unit">{unit}</span>}
    </span>
  );
}
