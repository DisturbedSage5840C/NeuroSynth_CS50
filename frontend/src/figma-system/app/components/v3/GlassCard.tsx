// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { ReactNode } from 'react';
import './v3.css';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  glow?: boolean;
}

export function GlassCard({ children, className = '', glow = false }: GlassCardProps) {
  return (
    <div className={`glass-card${glow ? ' glass-card-glow' : ''} ${className}`}>
      {children}
    </div>
  );
}
