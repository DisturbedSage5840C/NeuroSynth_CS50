// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { lazy, Suspense, useState } from 'react';
import { GlassCard } from './v3/GlassCard';
import { RiskChip } from './v3/RiskChip';
import { useMobile } from '@/hooks/useMobile';

// Heavy Three.js canvas — lazy loaded and only rendered on non-mobile
const BrainVisualization3D = lazy(() => import('./v2/BrainVisualization3D'));
// Lightweight SVG fallback for mobile / low-power devices (Part 8 risk mitigation)
const BrainVisualization2D = lazy(() => import('./v2/BrainVisualization2D'));

const REGIONS = [
  { id: 'frontal_l',  name: 'Frontal (L)',   disease: "Alzheimer's Disease", shapValue:  0.18 },
  { id: 'frontal_r',  name: 'Frontal (R)',   disease: "Alzheimer's Disease", shapValue:  0.15 },
  { id: 'parietal_l', name: 'Parietal (L)',  disease: "Alzheimer's Disease", shapValue:  0.09 },
  { id: 'parietal_r', name: 'Parietal (R)',  disease: "Alzheimer's Disease", shapValue:  0.07 },
  { id: 'temporal_l', name: 'Temporal (L)',  disease: "Parkinson's Disease", shapValue:  0.22 },
  { id: 'temporal_r', name: 'Temporal (R)',  disease: "Parkinson's Disease", shapValue:  0.19 },
  { id: 'occipital_l',name: 'Occipital (L)', disease: 'Multiple Sclerosis',  shapValue: -0.06 },
  { id: 'occipital_r',name: 'Occipital (R)', disease: 'Multiple Sclerosis',  shapValue: -0.05 },
  { id: 'cerebellum', name: 'Cerebellum',    disease: "Parkinson's Disease", shapValue:  0.14 },
  { id: 'brainstem',  name: 'Brainstem',     disease: 'ALS',                 shapValue:  0.31 },
];

const DISEASE_FILTER = [
  'All', "Alzheimer's Disease", "Parkinson's Disease", 'Multiple Sclerosis', 'ALS',
];

export function BrainAtlas() {
  const isMobile = useMobile();
  const [filter, setFilter] = useState('All');
  const [selectedRegion, setSelectedRegion] = useState<(typeof REGIONS)[0] | null>(null);

  const filteredRegions = filter === 'All'
    ? REGIONS
    : REGIONS.filter((r) => r.disease === filter);

  const shapValues = filteredRegions.map((r) => ({ feature: r.id, value: r.shapValue }));

  return (
    <div className="page-root">
      <div className="page-header">
        <h1 className="page-title">Brain Atlas</h1>
        <p className="page-subtitle">
          SHAP attribution mapped to anatomical brain regions — rotate to explore
        </p>
      </div>

      {/* Disease filter */}
      <div className="flex gap-2 flex-wrap mb-5">
        {DISEASE_FILTER.map((d) => (
          <button
            key={d}
            type="button"
            onClick={() => setFilter(d)}
            className={`traj-tab${filter === d ? ' traj-tab-active' : ''}`}
          >
            {d === 'All' ? 'All Diseases' : d.replace("'s Disease", '').replace(' Disease', '')}
          </button>
        ))}
      </div>

      <div className="page-grid-2">
        {/* Brain canvas — 3D on desktop, lightweight 2D SVG on mobile */}
        <GlassCard glow>
          <Suspense fallback={
            <div className="flex items-center justify-center h-72 text-sm text-muted-foreground">
              Loading brain atlas…
            </div>
          }>
            {isMobile
              ? <BrainVisualization2D shapValues={shapValues} focusDisease={filter === 'All' ? null : filter} />
              : <BrainVisualization3D shapValues={shapValues} focusDisease={filter === 'All' ? null : filter} />
            }
          </Suspense>
        </GlassCard>

        {/* Region list */}
        <GlassCard>
          <h3 className="text-sm font-medium text-foreground mb-3">
            {filteredRegions.length} Region{filteredRegions.length !== 1 ? 's' : ''}
            {filter !== 'All' && ` — ${filter.replace("'s Disease", '').replace(' Disease', '')}`}
          </h3>
          <div className="space-y-2 overflow-y-auto max-h-80 pr-1">
            {filteredRegions.map((r) => (
              <button
                key={r.id}
                type="button"
                className={`w-full text-left px-3 py-2.5 rounded-lg border transition-all ${
                  selectedRegion?.id === r.id
                    ? 'border-primary bg-primary/10'
                    : 'border-border hover:border-primary/40 hover:bg-secondary'
                }`}
                onClick={() => setSelectedRegion(selectedRegion?.id === r.id ? null : r)}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm text-foreground">{r.name}</span>
                  <span
                    className="text-xs font-mono"
                    style={{ color: r.shapValue > 0 ? 'var(--risk-high)' : 'var(--risk-low)' }}
                  >
                    {r.shapValue > 0 ? '+' : ''}{r.shapValue.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <RiskChip disease={r.disease} />
                </div>
              </button>
            ))}
          </div>

          {/* Selected region detail */}
          {selectedRegion && (
            <div className="mt-4 pt-4 border-t border-border">
              <div className="text-xs text-muted-foreground mb-1">Selected region</div>
              <div className="text-sm font-semibold text-foreground">{selectedRegion.name}</div>
              <div className="flex items-center gap-3 mt-2">
                <RiskChip disease={selectedRegion.disease} />
                <span className="text-xs text-muted-foreground">
                  SHAP: <span
                    className="font-mono"
                    style={{ color: selectedRegion.shapValue > 0 ? 'var(--risk-high)' : 'var(--risk-low)' }}
                  >
                    {selectedRegion.shapValue > 0 ? '+' : ''}{selectedRegion.shapValue.toFixed(4)}
                  </span>
                </span>
              </div>
              <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                This region shows {selectedRegion.shapValue > 0 ? 'elevated' : 'reduced'} SHAP attribution,
                indicating it {selectedRegion.shapValue > 0 ? 'increases' : 'decreases'} predicted risk
                for {selectedRegion.disease}.
              </p>
            </div>
          )}
        </GlassCard>
      </div>

      {/* Color legend */}
      <GlassCard className="mt-4">
        <div className="flex items-center gap-8 text-xs text-muted-foreground flex-wrap">
          <span className="font-medium text-foreground">SHAP color scale:</span>
          <div className="flex items-center gap-2">
            <div className="w-16 h-2 rounded-full" style={{ background: 'linear-gradient(to right, #10b981, #64748b)' }} />
            <span>Protective (negative)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-16 h-2 rounded-full" style={{ background: 'linear-gradient(to right, #64748b, #ef4444)' }} />
            <span>Risk-increasing (positive)</span>
          </div>
          <span className="ml-auto">Drag to rotate · Scroll to zoom</span>
        </div>
      </GlassCard>
    </div>
  );
}
