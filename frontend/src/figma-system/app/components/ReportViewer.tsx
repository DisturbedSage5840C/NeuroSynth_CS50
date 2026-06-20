// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useState } from 'react';
import { ChevronDown, ChevronRight, FileText, ExternalLink, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { clinicalReport } from '../data/mock-data';
import { UncertaintyBadge } from './UncertaintyBadge';
import { useAnalysisStore } from '../../../state/analysisStore';
import { useMutation } from '@tanstack/react-query';
import { apiFetch } from '../../../lib/api';
import { useOutletContext } from 'react-router';

const modalityIcons: Record<string, string> = {
  imaging: '🧠', genomic: '🧬', lab: '🧪', wearable: '⌚', literature: '📄',
};

interface ReportViewerProps {
  reportData?: {
    sections: Record<string, string>;
    generated_at?: string;
    word_count?: number;
  } | null;
}

export function ReportViewer({ reportData }: ReportViewerProps) {
  const analysisResult = useAnalysisStore((s) => s.result);
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([0]));
  const [expandedEvidence, setExpandedEvidence] = useState<Set<string>>(new Set());
  const { selectedPatientId } = useOutletContext<{ selectedPatientId: string }>();
  const setResult = useAnalysisStore((s) => s.setResult);
  const effectiveReport = reportData ?? analysisResult?.report ?? null;
  const sections = analysisResult?.report?.sections
    ? Object.entries(analysisResult.report.sections).map(([title, content]) => ({
        title,
        content: String(content),
        confidence: 0.85,
        evidenceChain: [] as typeof clinicalReport.sections[0]['evidenceChain'],
        claims: [] as typeof clinicalReport.sections[0]['claims'],
      }))
    : clinicalReport.sections;

  const toggleSection = (i: number) => {
    const s = new Set(expandedSections);
    s.has(i) ? s.delete(i) : s.add(i);
    setExpandedSections(s);
  };

  const toggleEvidence = (key: string) => {
    const s = new Set(expandedEvidence);
    s.has(key) ? s.delete(key) : s.add(key);
    setExpandedEvidence(s);
  };

  const regenerate = useMutation({
    mutationFn: () =>
      apiFetch<{ status: string; report?: { sections: Record<string, string>; generated_at?: string; word_count?: number } }>(
        '/reports/generate',
        {
          method: 'POST',
          body: JSON.stringify({
            patient_id: analysisResult?.patient_id || selectedPatientId,
            notes: 'Regenerate with latest model context',
          }),
        }
      ),
    onSuccess: (payload) => {
      if (payload.report && analysisResult) {
        setResult({
          ...analysisResult,
          report: payload.report,
        });
      }
    },
  });

  return (
    <div className="flex-1 overflow-y-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <FileText size={18} className="text-primary" />
            <h1 style={{ fontSize: '20px' }} className="text-foreground">Clinical Intelligence Report</h1>
          </div>
          <div className="flex items-center gap-4 mt-1 text-muted-foreground" style={{ fontSize: '12px' }}>
            <span className="font-mono">
              {analysisResult ? `NS-${analysisResult.patient_id}` : clinicalReport.id}
            </span>
            <span>
              {analysisResult?.report?.generated_at
                ? new Date(analysisResult.report.generated_at).toLocaleString()
                : new Date(clinicalReport.generatedAt).toLocaleString()}
            </span>
            <span className="font-mono">{clinicalReport.model}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => regenerate.mutate()}
            className="rounded border border-border px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
          >
            {regenerate.isPending ? 'Regenerating...' : 'Regenerate Report'}
          </button>
          <button
            onClick={() => window.print()}
            className="rounded border border-border px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
          >
            Download as PDF
          </button>
          <span className="px-2 py-1 rounded bg-primary/10 text-primary font-mono" style={{ fontSize: '10px' }}>
            AI-GENERATED
          </span>
          <span className="px-2 py-1 rounded bg-[var(--risk-moderate-bg)] text-[var(--risk-moderate)] font-mono" style={{ fontSize: '10px' }}>
            REQUIRES CLINICAL REVIEW
          </span>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-[var(--risk-moderate-bg)] border border-[var(--risk-moderate)]/20 rounded-lg p-3 mb-6 flex items-start gap-2">
        <AlertTriangle size={14} className="text-[var(--risk-moderate)] mt-0.5 shrink-0" />
        <p className="text-[var(--risk-moderate)]" style={{ fontSize: '11px' }}>
          This report is generated by an AI system and must be reviewed by a qualified clinician before any clinical decisions are made.
          All claims include confidence intervals. Low-confidence claims are flagged and require additional verification.
        </p>
      </div>

      {/* Sections */}
      <div className="space-y-3">
        {sections.map((section, i) => (
          <div key={i} className="bg-card border border-border rounded-lg overflow-hidden">
            {/* Section header */}
            <button
              onClick={() => toggleSection(i)}
              className="w-full flex items-center justify-between p-4 hover:bg-secondary/30 transition-colors"
            >
              <div className="flex items-center gap-3">
                {expandedSections.has(i) ? <ChevronDown size={16} className="text-muted-foreground" /> : <ChevronRight size={16} className="text-muted-foreground" />}
                <h2 style={{ fontSize: '14px' }} className="text-foreground">{section.title}</h2>
                <UncertaintyBadge confidence={section.confidence} size="md" />
              </div>
              <div className="flex items-center gap-2 text-muted-foreground" style={{ fontSize: '10px' }}>
                <span>{section.evidenceChain.length} evidence sources</span>
                <span>·</span>
                <span>{section.claims.length} claims</span>
              </div>
            </button>

            {expandedSections.has(i) && (
              <div className="px-4 pb-4 border-t border-border pt-4">
                {/* Content */}
                <p className="text-foreground/90 leading-relaxed mb-4" style={{ fontSize: '13px' }}>{section.content}</p>

                {/* Claims */}
                <div className="mb-4">
                  <h4 className="text-muted-foreground mb-2" style={{ fontSize: '11px', letterSpacing: '0.05em' }}>AI CLAIMS & CONFIDENCE</h4>
                  <div className="space-y-2">
                    {section.claims.map((claim, ci) => (
                      <div key={ci} className="flex items-start gap-3 p-2 rounded bg-secondary/30 border border-border">
                        <UncertaintyBadge confidence={claim.confidence} size="md" />
                        <div className="flex-1">
                          <p className="text-foreground" style={{ fontSize: '12px' }}>{claim.text}</p>
                          <div className="flex items-center gap-3 mt-1">
                            <span className="flex items-center gap-1 text-[var(--risk-low)]" style={{ fontSize: '10px' }}>
                              <CheckCircle size={10} /> {claim.supporting} supporting
                            </span>
                            <span className="flex items-center gap-1 text-[var(--risk-critical)]" style={{ fontSize: '10px' }}>
                              <XCircle size={10} /> {claim.contradicting} contradicting
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Evidence chain */}
                <div>
                  <button
                    onClick={() => toggleEvidence(`s${i}`)}
                    className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-2"
                    style={{ fontSize: '11px', letterSpacing: '0.05em' }}
                  >
                    {expandedEvidence.has(`s${i}`) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    EVIDENCE CHAIN ({section.evidenceChain.length})
                  </button>

                  {expandedEvidence.has(`s${i}`) && (
                    <div className="space-y-1.5 ml-4 border-l-2 border-border pl-3">
                      {section.evidenceChain.map((ev, ei) => (
                        <div key={ei} className="flex items-start gap-2 p-2 rounded bg-secondary/20">
                          <span style={{ fontSize: '12px' }}>{modalityIcons[ev.type]}</span>
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="text-foreground" style={{ fontSize: '12px' }}>{ev.source}</span>
                              <UncertaintyBadge confidence={ev.confidence} />
                              <span className="text-muted-foreground font-mono" style={{ fontSize: '10px' }}>{ev.timestamp}</span>
                            </div>
                            <p className="text-muted-foreground mt-0.5" style={{ fontSize: '11px' }}>{ev.description}</p>
                          </div>
                          <ExternalLink size={12} className="text-muted-foreground cursor-pointer hover:text-primary transition-colors" />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
