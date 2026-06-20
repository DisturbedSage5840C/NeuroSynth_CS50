// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { genomicRisks } from '../data/mock-data';
import { Dna } from 'lucide-react';
import type { AnalysisResult } from '../types/analysis';

interface GenomicHeatmapProps {
  analysisResult?: AnalysisResult | null;
}

export function GenomicHeatmap({ analysisResult }: GenomicHeatmapProps) {
  const data = analysisResult?.shap_values?.length
    ? analysisResult.shap_values.map((s, i) => ({
        x: i % 8,
        y: Math.floor(i / 8),
        z: Math.abs(s.value) * 500 + 60,
        gene: s.feature,
        variant: 'shap',
        risk: Math.min(1, Math.abs(s.value)),
        confidence: 0.85,
        pathway: 'Model attribution',
      }))
    : [];

  const riskColor = (risk: number) => {
    if (risk > 0.75) return 'var(--risk-critical)';
    if (risk > 0.5) return 'var(--risk-high)';
    if (risk > 0.25) return 'var(--risk-moderate)';
    return 'var(--risk-low)';
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.[0]) return null;
    const d = payload[0].payload;
    return (
      <div className="rounded-lg border border-border bg-card p-3 text-xs shadow-lg">
        <p className="font-medium text-foreground">{d.gene} · {d.variant}</p>
        <p className="text-muted-foreground">{d.pathway}</p>
        <p style={{ color: riskColor(d.risk) }}>Risk: {(d.risk * 100).toFixed(0)}%</p>
        <p className="text-muted-foreground">Confidence: {(d.confidence * 100).toFixed(0)}%</p>
      </div>
    );
  };

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">Genomic Risk Map</h3>
        <span className="text-xs text-muted-foreground">
          {analysisResult?.shap_values?.length
            ? `${analysisResult.shap_values.length} SHAP attributions`
            : `${genomicRisks.length} reference variants`}
        </span>
      </div>
      {!analysisResult ? (
        <div className="flex h-[220px] flex-col items-center justify-center gap-3">
          <div className="w-12 h-12 rounded-full border-2 border-dashed border-border flex items-center justify-center">
            <Dna size={20} className="text-muted-foreground" />
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-foreground">No genomic attribution yet</div>
            <div className="text-xs text-muted-foreground mt-0.5">Run analysis to generate SHAP-based genomic map</div>
          </div>
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
              <XAxis type="number" dataKey="x" hide />
              <YAxis type="number" dataKey="y" hide />
              <ZAxis type="number" dataKey="z" range={[40, 300]} />
              <Tooltip content={<CustomTooltip />} cursor={false} />
              <Scatter data={data}>
                {data.map((d, i) => (
                  <Cell key={i} fill={riskColor(d.risk)} fillOpacity={0.7} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
            {(['critical', 'high', 'moderate', 'low'] as const).map((l) => (
              <span key={l} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full" style={{ background: `var(--risk-${l})` }} />
                {l}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
