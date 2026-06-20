// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import './v3.css';

interface ClinicalInputProps {
  id: string;
  label: string;
  unit?: string;
  value: string | number;
  onChange: (v: string) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  rangeHint?: string;
  type?: 'number' | 'text';
}

export function ClinicalInput({
  id, label, unit, value, onChange,
  min, max, step = 1, placeholder, rangeHint, type = 'number',
}: ClinicalInputProps) {
  const numVal = typeof value === 'number' ? value : parseFloat(value as string);
  const inRange = (min == null || numVal >= min) && (max == null || numVal <= max);
  const showWarning = type === 'number' && !isNaN(numVal) && !inRange;

  return (
    <div className="ci-root">
      <label className="ci-label" htmlFor={id}>{label}</label>
      <div className="ci-input-wrap">
        <input
          id={id}
          type={type}
          className={`ci-input${showWarning ? ' ci-input-warn' : ''}`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          min={min}
          max={max}
          step={step}
          placeholder={placeholder}
        />
        {unit && <span className="ci-unit">{unit}</span>}
      </div>
      {rangeHint && (
        <span className={`ci-hint${showWarning ? ' ci-hint-warn' : ''}`}>
          {showWarning ? `Out of range (${min}–${max})` : rangeHint}
        </span>
      )}
    </div>
  );
}
