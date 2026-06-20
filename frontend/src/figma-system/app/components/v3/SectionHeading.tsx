// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { ReactNode } from 'react';
import './v3.css';

interface SectionHeadingProps {
  children: ReactNode;
  action?: ReactNode;
}

export function SectionHeading({ children, action }: SectionHeadingProps) {
  return (
    <div className="sh-root">
      <span className="sh-label">{children}</span>
      <span className="sh-rule" aria-hidden />
      {action && <span className="sh-action">{action}</span>}
    </div>
  );
}
