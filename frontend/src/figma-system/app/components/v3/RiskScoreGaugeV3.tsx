// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useMemo } from 'react';
import { motion } from 'framer-motion';
import '../v3/v3.css';

interface Props {
  probability: number;
  riskLevel: string;
  confidence: string;
  diseaseProbabilities?: Record<string, number>;
}

const CX = 130;
const CY = 110;
const R_OUTER = 90;
const R_INNER = 62;
const NEEDLE_LEN = 78;

// Map fraction (0→1) to SVG point on the semicircle arc.
// Arc runs left (0%) → top → right (100%).
function arcPoint(frac: number, r: number): [number, number] {
  const angle = Math.PI * (1 - frac); // π → 0
  return [
    CX + r * Math.cos(angle),
    CY - r * Math.sin(angle),
  ];
}

function arcPath(startFrac: number, endFrac: number): string {
  const [x0, y0] = arcPoint(startFrac, R_OUTER);
  const [x1, y1] = arcPoint(endFrac,   R_OUTER);
  const [xi0, yi0] = arcPoint(startFrac, R_INNER);
  const [xi1, yi1] = arcPoint(endFrac,   R_INNER);
  const large = endFrac - startFrac > 0.5 ? 1 : 0;
  return [
    `M ${x0} ${y0}`,
    `A ${R_OUTER} ${R_OUTER} 0 ${large} 1 ${x1} ${y1}`,
    `L ${xi1} ${yi1}`,
    `A ${R_INNER} ${R_INNER} 0 ${large} 0 ${xi0} ${yi0}`,
    'Z',
  ].join(' ');
}

const ZONES = [
  { startFrac: 0,    endFrac: 0.40, color: '#10b981', label: 'Low'      },
  { startFrac: 0.40, endFrac: 0.65, color: '#f59e0b', label: 'Moderate' },
  { startFrac: 0.65, endFrac: 1.00, color: '#ef4444', label: 'High'     },
];

const DISEASE_COLORS: Record<string, string> = {
  "Alzheimer's Disease": '#818cf8',
  "Parkinson's Disease": '#34d399',
  'Multiple Sclerosis':  '#fb923c',
  'Epilepsy':            '#a78bfa',
  'ALS':                 '#f87171',
  "Huntington's Disease":'#fbbf24',
};

export function RiskScoreGaugeV3({ probability, riskLevel, confidence, diseaseProbabilities }: Props) {
  const pct = Math.round(probability * 100);

  const needleColor = useMemo(() => {
    const level = riskLevel.toLowerCase();
    if (level.includes('critical') || level.includes('high')) return '#ef4444';
    if (level.includes('moderate')) return '#f59e0b';
    return '#10b981';
  }, [riskLevel]);

  // Needle endpoint (animated from left-start to final position)
  const [nx, ny] = arcPoint(probability, NEEDLE_LEN);
  const [nx0, ny0] = arcPoint(0, NEEDLE_LEN);

  const topDiseases = useMemo(() => {
    if (!diseaseProbabilities) return [];
    return Object.entries(diseaseProbabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4);
  }, [diseaseProbabilities]);

  return (
    <div className="gauge-wrap">
      <svg
        className="gauge-svg"
        width={CX * 2}
        height={CY + 28}
        viewBox={`0 0 ${CX * 2} ${CY + 28}`}
        aria-label={`Risk ${pct}% — ${riskLevel}`}
      >
        {/* Track */}
        {ZONES.map((z) => (
          <path
            key={z.label}
            d={arcPath(z.startFrac, z.endFrac)}
            fill={z.color}
            opacity={0.15}
          />
        ))}

        {/* Filled arc up to probability */}
        {probability > 0 && (
          <motion.path
            d={arcPath(0, probability)}
            fill={needleColor}
            opacity={0.55}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.55 }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        )}

        {/* Zone tick marks */}
        {[0.40, 0.65].map((f) => {
          const [tx, ty] = arcPoint(f, R_OUTER + 4);
          const [tx2, ty2] = arcPoint(f, R_INNER - 4);
          return (
            <line
              key={f}
              x1={tx} y1={ty} x2={tx2} y2={ty2}
              stroke="var(--bg-base)" strokeWidth={2}
            />
          );
        })}

        {/* Needle */}
        <motion.line
          x1={CX} y1={CY}
          x2={nx0} y2={ny0}
          stroke={needleColor}
          strokeWidth={2.5}
          strokeLinecap="round"
          animate={{ x2: nx, y2: ny }}
          transition={{ type: 'spring', damping: 14, stiffness: 90, delay: 0.3 }}
        />
        <circle cx={CX} cy={CY} r={5} fill={needleColor} />
        <circle cx={CX} cy={CY} r={2.5} fill="var(--bg-base)" />

        {/* Center label */}
        <text x={CX} y={CY + 22} className="gauge-center gauge-pct" fill={needleColor} fontSize={26}>
          {pct}%
        </text>
        <text x={CX} y={CY + 38} className="gauge-center gauge-level" fill={needleColor} fontSize={8}>
          {riskLevel.toUpperCase()}
        </text>

        {/* Zone labels */}
        <text x={arcPoint(0.20, R_OUTER + 14)[0]} y={arcPoint(0.20, R_OUTER + 14)[1]}
          textAnchor="middle" fontSize={7} fill="#10b981" opacity={0.7}>Low</text>
        <text x={arcPoint(0.52, R_OUTER + 14)[0]} y={arcPoint(0.52, R_OUTER + 14)[1]}
          textAnchor="middle" fontSize={7} fill="#f59e0b" opacity={0.7}>Mod</text>
        <text x={arcPoint(0.82, R_OUTER + 14)[0]} y={arcPoint(0.82, R_OUTER + 14)[1]}
          textAnchor="middle" fontSize={7} fill="#ef4444" opacity={0.7}>High</text>
      </svg>

      {/* Confidence + model count */}
      <div className="gauge-meta">
        <span>
          <span
            className="gauge-meta-dot"
            style={{
              backgroundColor:
                confidence === 'High' ? '#22c55e'
                : confidence === 'Medium' ? '#eab308'
                : '#ef4444',
            }}
          />
          {confidence} confidence
        </span>
      </div>

      {/* Per-disease probability bars */}
      {topDiseases.length > 0 && (
        <div style={{ width: '100%', maxWidth: 220 }}>
          {topDiseases.map(([name, prob]) => (
            <div key={name} style={{ marginBottom: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ fontSize: 9, color: 'var(--text-tertiary)' }}>
                  {name.replace("'s Disease", '').replace(' Disease', '')}
                </span>
                <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {Math.round(Number(prob) * 100)}%
                </span>
              </div>
              <div style={{ height: 3, borderRadius: 2, background: 'var(--bg-subtle)' }}>
                <motion.div
                  style={{
                    height: '100%',
                    borderRadius: 2,
                    background: DISEASE_COLORS[name] ?? 'var(--accent-primary)',
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.round(Number(prob) * 100)}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut', delay: 0.4 }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
