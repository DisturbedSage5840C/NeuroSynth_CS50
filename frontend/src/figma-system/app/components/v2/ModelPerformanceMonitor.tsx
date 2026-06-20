// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { motion } from 'framer-motion';
import { Activity, CheckCircle2, XCircle, AlertTriangle, Shield } from 'lucide-react';

interface GateStatus {
  name: string;
  type: 'hard' | 'soft';
  result: 'PASS' | 'HARD_FAIL' | 'SOFT_WARN';
  metric: string;
  value: number;
  threshold: number;
}

interface ModelPerformanceMonitorProps {
  auc?: number;
  ece?: number;
  f1?: number;
  brier?: number;
  decision?: 'PROMOTE' | 'REJECT' | 'HUMAN_REVIEW';
  gates?: GateStatus[];
  modelVersion?: string;
}

const DECISION_STYLES: Record<string, { color: string; bg: string; icon: typeof CheckCircle2 }> = {
  PROMOTE: { color: '#22c55e', bg: 'rgba(34,197,94,0.1)', icon: CheckCircle2 },
  REJECT: { color: '#ef4444', bg: 'rgba(239,68,68,0.1)', icon: XCircle },
  HUMAN_REVIEW: { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', icon: AlertTriangle },
};

export function ModelPerformanceMonitor({
  auc,
  ece,
  f1,
  brier,
  decision,
  gates = [],
  modelVersion = 'v2.0.0',
}: ModelPerformanceMonitorProps) {
  const hasData = decision !== undefined && decision !== null;

  const ds = DECISION_STYLES[decision ?? 'REJECT'] || DECISION_STYLES.REJECT;
  const DecisionIcon = ds.icon;

  const metrics = [
    { label: 'AUC', value: auc, target: 0.80, format: (v: number) => v.toFixed(4) },
    { label: 'F1', value: f1, target: 0.75, format: (v: number) => v.toFixed(4) },
    { label: 'ECE', value: ece, target: 0.05, format: (v: number) => v.toFixed(4), invert: true },
    { label: 'Brier', value: brier, target: 0.15, format: (v: number) => v.toFixed(4), invert: true },
  ].filter(m => m.value !== undefined) as Array<{ label: string; value: number; target: number; format: (v: number) => string; invert?: boolean }>;

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-primary" />
          <h3 className="text-sm font-medium text-foreground">Model Performance Monitor</h3>
        </div>
        <span className="text-xs font-mono text-muted-foreground">{modelVersion}</span>
      </div>

      {!hasData ? (
        <div className="flex flex-col items-center justify-center py-8 gap-2 text-muted-foreground">
          <Activity size={24} className="opacity-30" />
          <span className="text-xs">Run analysis to see model performance metrics</span>
        </div>
      ) : (
        <>
          {/* Decision badge */}
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="flex items-center gap-3 p-3 rounded-lg mb-4"
            style={{ background: ds.bg, border: `1px solid ${ds.color}30` }}
          >
            <DecisionIcon size={20} style={{ color: ds.color }} />
            <div>
              <div className="text-sm font-semibold" style={{ color: ds.color }}>
                {decision}
              </div>
              <div className="text-xs text-muted-foreground">
                {decision === 'PROMOTE' && 'All gates passed — production-ready'}
                {decision === 'REJECT' && 'Hard gate failures — return to training'}
                {decision === 'HUMAN_REVIEW' && 'Soft warnings — requires sign-off'}
              </div>
            </div>
          </motion.div>

          {/* Metrics grid */}
          <div className="grid grid-cols-4 gap-3 mb-4">
            {metrics.map((m, i) => {
              const passing = m.invert ? m.value <= m.target : m.value >= m.target;
              return (
                <motion.div
                  key={m.label}
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: i * 0.08 }}
                  className="rounded-lg p-3 text-center"
                  style={{ background: 'var(--card-elevated)', border: '1px solid var(--border)' }}
                >
                  <div className="text-xs text-muted-foreground mb-1">{m.label}</div>
                  <div
                    className="text-lg font-mono font-bold"
                    style={{ color: passing ? '#22c55e' : '#ef4444' }}
                  >
                    {m.format(m.value)}
                  </div>
                  <div className="text-[10px] text-muted-foreground mt-0.5">
                    {m.invert ? '≤' : '≥'} {m.target}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </>
      )}

      {/* Gate results */}
      {gates.length > 0 && (
        <div className="space-y-1.5">
          <div className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1.5">
            <Shield size={12} /> Validation Gates
          </div>
          {gates.map((gate, i) => {
            const passed = gate.result === 'PASS';
            const warn = gate.result === 'SOFT_WARN';
            return (
              <motion.div
                key={gate.name}
                initial={{ x: -8, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: i * 0.05 }}
                className="flex items-center justify-between py-1.5 px-2 rounded text-xs"
                style={{ background: i % 2 === 0 ? 'var(--card-elevated)' : 'transparent' }}
              >
                <div className="flex items-center gap-2">
                  {passed ? (
                    <CheckCircle2 size={12} className="text-green-400" />
                  ) : warn ? (
                    <AlertTriangle size={12} className="text-amber-400" />
                  ) : (
                    <XCircle size={12} className="text-red-400" />
                  )}
                  <span className="text-foreground">{gate.name}</span>
                  <span className="text-muted-foreground font-mono">
                    [{gate.type}]
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-mono text-muted-foreground">
                    {gate.metric}={gate.value.toFixed(4)}
                  </span>
                  <span
                    className="font-mono font-semibold"
                    style={{ color: passed ? '#22c55e' : warn ? '#f59e0b' : '#ef4444' }}
                  >
                    {gate.result}
                  </span>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
