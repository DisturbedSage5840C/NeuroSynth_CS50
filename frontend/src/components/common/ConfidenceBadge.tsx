// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
type ConfidenceTone = "high" | "medium" | "low";

export interface ConfidenceBadgeProps {
  confidence: number;
}

function tone(confidence: number): ConfidenceTone {
  if (confidence >= 0.8) return "high";
  if (confidence >= 0.5) return "medium";
  return "low";
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps): JSX.Element {
  const t = tone(confidence);
  const label = `${Math.round(confidence * 100)}%`;

  const cls =
    t === "high"
      ? "bg-success/20 text-success"
      : t === "medium"
        ? "bg-warning/20 text-warning"
        : "bg-danger/20 text-danger";

  return <span className={`rounded-full px-2 py-1 text-xs font-semibold ${cls}`}>{label}</span>;
}
