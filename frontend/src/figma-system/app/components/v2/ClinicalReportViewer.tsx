// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Download, Share2, ClipboardList, Stethoscope, Brain, CalendarCheck } from 'lucide-react';

interface SOAPReport {
  format: string;
  soap: {
    subjective: string;
    objective: string;
    assessment: string;
    plan: string;
  };
  icd10_codes: Array<{ code: string; description: string; confidence: number }>;
  generated_at: string;
  report_id: string;
  word_count: number;
  patient_id: string;
}

interface ClinicalReportViewerProps {
  report: SOAPReport | null;
  onExportPDF?: () => void;
  onExportFHIR?: () => void;
}

const SOAP_ICONS: Record<string, typeof ClipboardList> = {
  subjective: ClipboardList,
  objective: Stethoscope,
  assessment: Brain,
  plan: CalendarCheck,
};

const SOAP_COLORS: Record<string, string> = {
  subjective: '#818cf8',
  objective: '#34d399',
  assessment: '#f59e0b',
  plan: '#06b6d4',
};

const SOAP_LABELS: Record<string, string> = {
  subjective: 'Subjective',
  objective: 'Objective',
  assessment: 'Assessment',
  plan: 'Plan',
};

export function ClinicalReportViewer({ report, onExportPDF, onExportFHIR }: ClinicalReportViewerProps) {
  const [activeTab, setActiveTab] = useState<'soap' | 'icd10'>('soap');

  if (!report) {
    return (
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="text-sm font-medium text-foreground mb-3">Clinical Report</h3>
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <FileText size={28} className="mb-3 opacity-30" />
          <span className="text-sm">No report generated yet</span>
          <span className="text-xs mt-1">Run analysis to generate a SOAP-structured clinical report</span>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-border" style={{ background: 'var(--card-elevated)' }}>
        <div className="flex items-center gap-2">
          <FileText size={16} className="text-primary" />
          <h3 className="text-sm font-medium text-foreground">Clinical Report</h3>
          <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: 'var(--secondary)', color: 'var(--primary)' }}>
            {report.format}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {onExportPDF && (
            <button
              onClick={onExportPDF}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border border-border hover:border-primary hover:text-primary transition-all"
            >
              <Download size={12} /> PDF
            </button>
          )}
          {onExportFHIR && (
            <button
              onClick={onExportFHIR}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border border-border hover:border-primary hover:text-primary transition-all"
            >
              <Share2 size={12} /> FHIR R4
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border">
        {(['soap', 'icd10'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className="flex-1 text-xs font-medium py-2.5 transition-colors relative"
            style={{
              color: activeTab === tab ? 'var(--primary)' : 'var(--muted-foreground)',
            }}
          >
            {tab === 'soap' ? 'SOAP Report' : `ICD-10 Codes (${report.icd10_codes.length})`}
            {activeTab === tab && (
              <motion.div
                layoutId="report-tab"
                className="absolute bottom-0 left-0 right-0 h-0.5"
                style={{ backgroundColor: 'var(--primary)' }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-5">
        <AnimatePresence mode="wait">
          {activeTab === 'soap' ? (
            <motion.div
              key="soap"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="space-y-4"
            >
              {Object.entries(report.soap).map(([key, text], i) => {
                const Icon = SOAP_ICONS[key] || ClipboardList;
                const color = SOAP_COLORS[key] || 'var(--primary)';

                return (
                  <motion.div
                    key={key}
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="rounded-lg border border-border p-4"
                    style={{ borderLeftColor: color, borderLeftWidth: 3 }}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Icon size={14} style={{ color }} />
                      <span className="text-xs font-semibold tracking-wider" style={{ color }}>
                        {SOAP_LABELS[key] || key}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">{text}</p>
                  </motion.div>
                );
              })}
            </motion.div>
          ) : (
            <motion.div
              key="icd10"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
            >
              <div className="space-y-2">
                {report.icd10_codes.map((icd, i) => (
                  <motion.div
                    key={icd.code}
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.08 }}
                    className="flex items-center justify-between rounded-lg border border-border p-3"
                    style={{ background: 'var(--card-elevated)' }}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-mono font-bold text-primary">{icd.code}</span>
                      <span className="text-xs text-muted-foreground">{icd.description}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--secondary)' }}>
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${icd.confidence * 100}%` }}
                          transition={{ delay: i * 0.08 + 0.2, duration: 0.5 }}
                          className="h-full rounded-full"
                          style={{ backgroundColor: icd.confidence > 0.7 ? '#22c55e' : icd.confidence > 0.4 ? '#eab308' : '#6b7280' }}
                        />
                      </div>
                      <span className="text-xs font-mono" style={{ color: icd.confidence > 0.7 ? '#22c55e' : 'var(--muted-foreground)' }}>
                        {Math.round(icd.confidence * 100)}%
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-5 py-2.5 border-t border-border text-xs text-muted-foreground font-mono">
        <span>{report.report_id}</span>
        <span>{report.word_count} words</span>
        <span>{new Date(report.generated_at).toLocaleString()}</span>
      </div>
    </div>
  );
}
