// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import './v3.css';

const DISEASE_CLASS: Record<string, string> = {
  "Alzheimer's Disease": 'risk-chip-ad',
  "Parkinson's Disease": 'risk-chip-pd',
  'Multiple Sclerosis':  'risk-chip-ms',
  'Epilepsy':            'risk-chip-ep',
  'ALS':                 'risk-chip-als',
  "Huntington's Disease":'risk-chip-hd',
};

interface RiskChipProps {
  disease: string;
  probability?: number;
}

export function RiskChip({ disease, probability }: RiskChipProps) {
  const cls = DISEASE_CLASS[disease] ?? 'risk-chip-default';
  const abbrev = disease.replace("'s Disease", '').replace(' Disease', '').split(' ').pop();
  return (
    <span className={`risk-chip ${cls}`}>
      {abbrev}
      {probability !== undefined && (
        <span className="risk-chip-prob">{Math.round(probability * 100)}%</span>
      )}
    </span>
  );
}
