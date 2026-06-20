// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { ReactNode } from 'react';
import './v3.css';

interface TimelineItemProps {
  date: string;
  title: string;
  description?: string;
  badge?: ReactNode;
  last?: boolean;
}

export function TimelineItem({ date, title, description, badge, last = false }: TimelineItemProps) {
  return (
    <div className={`tl-item${last ? ' tl-item-last' : ''}`}>
      <div className="tl-line-col">
        <span className="tl-dot" />
        {!last && <span className="tl-line" />}
      </div>
      <div className="tl-content">
        <div className="tl-header">
          <span className="tl-date">{date}</span>
          {badge && <span className="tl-badge">{badge}</span>}
        </div>
        <div className="tl-title">{title}</div>
        {description && <div className="tl-desc">{description}</div>}
      </div>
    </div>
  );
}
