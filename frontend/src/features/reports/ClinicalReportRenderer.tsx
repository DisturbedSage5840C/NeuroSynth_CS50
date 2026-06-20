// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { ClinicalReportPayload } from "@/types/api";
import { ConfidenceBadge } from "@/components/common/ConfidenceBadge";

export interface ClinicalReportRendererProps {
  report: ClinicalReportPayload;
}

export function ClinicalReportRenderer({ report }: ClinicalReportRendererProps): JSX.Element {
  return (
    <article className="space-y-4 text-sm" aria-label="Clinical report">
      <section>
        <h3 className="text-base font-semibold">Assessment</h3>
        <p className="mt-1 text-muted">{report.assessment}</p>
        <p className="mt-2 inline-flex items-center gap-2">
          <span className="font-semibold uppercase">Risk: {report.risk_level}</span>
          <ConfidenceBadge confidence={Math.max(0.2, Math.min(0.95, report.risk_level === "critical" ? 0.92 : report.risk_level === "high" ? 0.78 : report.risk_level === "moderate" ? 0.56 : 0.32))} />
        </p>
      </section>

      <section>
        <h3 className="text-base font-semibold">Causal pathways</h3>
        <ul className="mt-2 space-y-2">
          {report.causal_pathways.map((p) => (
            <li key={p.pathway} className="rounded border border-line bg-surface/60 p-2">
              <div className="flex items-center justify-between">
                <strong>{p.pathway}</strong>
                <ConfidenceBadge confidence={p.confidence} />
              </div>
              <p className="text-muted">{p.rationale}</p>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3 className="text-base font-semibold">Interventions</h3>
        <ol className="mt-2 space-y-2">
          {report.interventions.map((i) => (
            <li key={`${i.name}-${i.priority}`} className="rounded border border-line bg-surface/60 p-2">
              <div className="flex items-center justify-between">
                <strong>{i.priority}. {i.name}</strong>
                <ConfidenceBadge confidence={Math.max(0.2, Math.min(0.95, i.confidence ?? 0.66))} />
              </div>
              <p className="text-muted">{i.action}</p>
              <p className="text-muted">Expected impact: {i.expected_impact}</p>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h3 className="text-base font-semibold">Evidence chain</h3>
        {report.evidence_refs.map((ref) => (
          <details key={ref} className="mt-2 rounded border border-line p-2">
            <summary className="cursor-pointer">{ref}</summary>
            <p className="mt-1 text-muted">Resolved evidence reference from hybrid retrieval context.</p>
          </details>
        ))}
      </section>

      <section>
        <h3 className="text-base font-semibold">Uncertainty note</h3>
        <p className="text-muted">{report.uncertainty_note}</p>
      </section>

      <div className="print:hidden">
        <button className="rounded bg-primary px-3 py-2 text-xs font-semibold text-surface" onClick={() => window.print()}>
          Export PDF
        </button>
      </div>
    </article>
  );
}
