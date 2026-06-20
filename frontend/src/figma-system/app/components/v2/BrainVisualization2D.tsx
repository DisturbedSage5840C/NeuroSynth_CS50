// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
/**
 * BrainVisualization2D — lightweight SVG fallback for mobile/low-power devices.
 *
 * Renders all 116 AAL regions as colored circles on a schematic brain outline
 * (coronal top-down projection). Respects disease-focus dimming and SHAP color
 * coding, with zero Three.js / WebGL dependency.
 */
import { useState, useMemo } from 'react';
import { AAL_REGIONS, mapFeatureToAAL, type AALRegion } from '@/lib/aalAtlas';

interface SHAPValue { feature: string; value: number; }

interface Props {
  shapValues?: SHAPValue[];
  focusDisease?: string | null;
  height?: number;
}

function buildShapMap(shapValues: SHAPValue[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const sv of shapValues) {
    const id = mapFeatureToAAL(sv.feature);
    m.set(id, (m.get(id) ?? 0) + sv.value);
  }
  return m;
}

function shapColor(v: number, dimmed: boolean): string {
  if (dimmed) return '#1c2a42';
  if (v > 0.05) return `hsl(0 80% ${30 + Math.min(v / 0.3, 1) * 30}%)`;
  if (v < -0.05) return `hsl(150 60% ${30 + Math.min(-v / 0.3, 1) * 30}%)`;
  return '#2a3a52';
}

// Project 3-D MNI [x,y,z] → 2-D SVG [px, py].
// Use axial (top-down) projection: px = x (LR), py = -z (AP)
function project(pos: [number, number, number], w: number, h: number): [number, number] {
  const [x, , z] = pos;
  const px = w / 2 + x * (w / 4);
  const py = h / 2 - z * (h / 4);
  return [px, py];
}

export default function BrainVisualization2D({
  shapValues = [],
  focusDisease = null,
  height = 320,
}: Props) {
  const [tooltip, setTooltip] = useState<{ region: AALRegion; sv: number; x: number; y: number } | null>(null);
  const W = 400;
  const H = height;

  const shapMap = useMemo(() => buildShapMap(shapValues), [shapValues]);

  return (
    <div className="brain2d-root" style={{ height: H }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        height={H}
        className="brain2d-svg"
        aria-label="2D brain atlas — axial projection"
      >
        {/* Schematic brain outline (ellipse approximation) */}
        <ellipse
          cx={W / 2} cy={H / 2}
          rx={W * 0.44} ry={H * 0.42}
          fill="none"
          stroke="rgba(0,212,255,0.15)"
          strokeWidth={1.5}
        />
        {/* Midline */}
        <line
          x1={W / 2} y1={H * 0.1}
          x2={W / 2} y2={H * 0.9}
          stroke="rgba(0,212,255,0.08)"
          strokeWidth={1}
          strokeDasharray="4 4"
        />

        {/* Region dots */}
        {AAL_REGIONS.map((r) => {
          const sv = shapMap.get(r.id) ?? 0;
          const dimmed = focusDisease != null
            ? !(r.diseases ?? []).includes(focusDisease)
            : false;
          const [px, py] = project(r.position, W, H);
          const radius = r.scale * 55;
          const color = shapColor(sv, dimmed);
          return (
            <circle
              key={r.id}
              cx={px} cy={py} r={Math.max(2, radius)}
              fill={color}
              fillOpacity={dimmed ? 0.3 : 0.75}
              stroke={dimmed ? 'transparent' : 'rgba(0,212,255,0.2)'}
              strokeWidth={0.5}
              className="brain2d-region"
              onMouseEnter={() => setTooltip({ region: r, sv, x: px, y: py })}
              onMouseLeave={() => setTooltip(null)}
            />
          );
        })}

        {/* Tooltip */}
        {tooltip && (
          <g>
            <rect
              x={Math.min(tooltip.x + 6, W - 120)}
              y={Math.max(tooltip.y - 28, 4)}
              width={114} height={26}
              rx={4}
              fill="rgba(6,11,24,0.92)"
              stroke="rgba(0,212,255,0.2)"
            />
            <text
              x={Math.min(tooltip.x + 12, W - 114)}
              y={Math.max(tooltip.y - 10, 18)}
              fontSize={8}
              fill="#94a3b8"
              fontFamily="var(--font-mono)"
            >
              {tooltip.region.name.slice(0, 20)}
              {tooltip.sv !== 0 && (
                <tspan fill={tooltip.sv > 0 ? '#f87171' : '#34d399'} dx={4}>
                  {tooltip.sv > 0 ? '+' : ''}{tooltip.sv.toFixed(3)}
                </tspan>
              )}
            </text>
          </g>
        )}
      </svg>

      {/* Legend */}
      <div className="brain2d-legend">
        <span className="brain2d-leg-item brain2d-risk">risk</span>
        <span className="brain2d-leg-item brain2d-protect">protective</span>
        <span className="brain2d-leg-label">Axial view · {AAL_REGIONS.length} regions</span>
      </div>
    </div>
  );
}
