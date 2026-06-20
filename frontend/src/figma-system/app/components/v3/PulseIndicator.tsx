// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import './v3.css';

type Status = 'live' | 'idle' | 'error';

interface PulseIndicatorProps {
  status?: Status;
  label?: string;
}

const STATUS_CLASS: Record<Status, string> = {
  live:  'pulse-live',
  idle:  'pulse-idle',
  error: 'pulse-error',
};

export function PulseIndicator({ status = 'live', label }: PulseIndicatorProps) {
  return (
    <span className="pulse-root">
      <span className={`pulse-dot ${STATUS_CLASS[status]}`} aria-hidden />
      {label && <span className="pulse-label">{label}</span>}
    </span>
  );
}
