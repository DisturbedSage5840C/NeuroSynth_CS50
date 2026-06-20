// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { PropsWithChildren } from "react";

interface PanelCardProps extends PropsWithChildren {
  title: string;
  subtitle?: string;
}

export function PanelCard({ title, subtitle, children }: PanelCardProps): JSX.Element {
  return (
    <section className="rounded-panel border border-line bg-elevated/80 p-4 shadow-panel">
      <header className="mb-3">
        <h2 className="text-lg font-semibold text-text">{title}</h2>
        {subtitle ? <p className="text-sm text-muted">{subtitle}</p> : null}
      </header>
      {children}
    </section>
  );
}
