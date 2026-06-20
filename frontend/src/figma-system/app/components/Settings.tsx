// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useState } from 'react';
import { Save, Key, Palette, Sliders, Info } from 'lucide-react';
import { GlassCard } from './v3/GlassCard';

interface SettingRowProps {
  label: string;
  description: string;
  children: React.ReactNode;
}

function SettingRow({ label, description, children }: SettingRowProps) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-border last:border-0">
      <div className="min-w-0 mr-4">
        <div className="text-sm font-medium text-foreground">{label}</div>
        <div className="text-xs text-muted-foreground mt-0.5">{description}</div>
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className="w-10 h-5 rounded-full relative transition-colors"
      style={{ background: value ? 'var(--accent-primary)' : 'var(--bg-subtle)' }}
      aria-checked={value}
      role="switch"
    >
      <span
        className="absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform"
        style={{ transform: value ? 'translateX(21px)' : 'translateX(2px)' }}
      />
    </button>
  );
}

export function Settings() {
  const [apiKey, setApiKey]           = useState(() => localStorage.getItem('ns_openai_key') ?? '');
  const [ragEnabled, setRagEnabled]   = useState(() => localStorage.getItem('ns_rag') !== 'false');
  const [tabnet, setTabnet]           = useState(() => localStorage.getItem('ns_tabnet') !== 'false');
  const [showVariance, setShowVariance] = useState(() => localStorage.getItem('ns_variance') === 'true');
  const [saved, setSaved]             = useState(false);

  const save = () => {
    localStorage.setItem('ns_openai_key', apiKey);
    localStorage.setItem('ns_rag',        String(ragEnabled));
    localStorage.setItem('ns_tabnet',     String(tabnet));
    localStorage.setItem('ns_variance',   String(showVariance));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="page-root">
      <div className="page-header flex items-center justify-between">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-subtitle">API keys, feature flags, and display preferences</p>
        </div>
        <button
          type="button"
          onClick={save}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:opacity-90 transition-opacity"
        >
          <Save size={14} />
          {saved ? 'Saved!' : 'Save'}
        </button>
      </div>

      <div className="space-y-4 max-w-2xl">
        {/* API keys */}
        <GlassCard>
          <div className="flex items-center gap-2 mb-4">
            <Key size={15} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">API Keys</h3>
          </div>
          <SettingRow
            label="OpenAI API Key"
            description="Required for RAG literature search and corpus embedding"
          >
            <input
              type="password"
              className="w-52 bg-transparent border border-border rounded-lg px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary transition-colors"
              placeholder="sk-…"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </SettingRow>
        </GlassCard>

        {/* Feature flags */}
        <GlassCard>
          <div className="flex items-center gap-2 mb-4">
            <Sliders size={15} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Feature Flags</h3>
          </div>
          <SettingRow
            label="RAG Literature Search"
            description="Attach PubMed abstracts to SOAP reports as citations"
          >
            <Toggle value={ragEnabled} onChange={setRagEnabled} />
          </SettingRow>
          <SettingRow
            label="TabNet (6th base learner)"
            description="Slower but adds attention-based tabular interpretability"
          >
            <Toggle value={tabnet} onChange={setTabnet} />
          </SettingRow>
          <SettingRow
            label="Ensemble Variance display"
            description="Show model disagreement score in risk gauge"
          >
            <Toggle value={showVariance} onChange={setShowVariance} />
          </SettingRow>
        </GlassCard>

        {/* Theme info */}
        <GlassCard>
          <div className="flex items-center gap-2 mb-4">
            <Palette size={15} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Design System</h3>
          </div>
          <div className="grid grid-cols-6 gap-3 mb-4">
            {[
              { label: 'Accent',     color: 'var(--accent-primary)' },
              { label: "Alzheimer's",color: 'var(--color-ad)' },
              { label: "Parkinson's",color: 'var(--color-pd)' },
              { label: 'MS',         color: 'var(--color-ms)' },
              { label: 'ALS',        color: 'var(--color-als)' },
              { label: "Huntington's",color:'var(--color-hd)' },
            ].map(({ label, color }) => (
              <div key={label} className="flex flex-col items-center gap-1.5">
                <div className="w-8 h-8 rounded-lg border border-border" style={{ background: color }} />
                <span className="text-xs text-muted-foreground text-center leading-tight">{label}</span>
              </div>
            ))}
          </div>
          <div className="flex items-start gap-2 text-xs text-muted-foreground">
            <Info size={12} className="flex-shrink-0 mt-0.5" />
            Neural Interface theme — deep space dark with electric cyan accent. Typography: Inter + JetBrains Mono.
          </div>
        </GlassCard>

        {/* Version info */}
        <GlassCard>
          <h3 className="text-sm font-semibold text-foreground mb-3">Version</h3>
          {[
            ['Platform',      'NeuroSynth v5.0'],
            ['ML Ensemble',   'RF + GB + CatBoost + LR + LightGBM (+ TabNet opt.)'],
            ['Fusion',        'CrossAttentionFusion (Optuna-tuned weights)'],
            ['Training AUC',  '0.9940 (real patient data)'],
            ['Data',          '16,026 rows · 56 features · 11 sources'],
            ['RAG',           '10,000 PubMed abstracts (pgvector / Neon)'],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between py-1.5 text-xs border-b border-border last:border-0">
              <span className="text-muted-foreground">{k}</span>
              <span className="text-foreground font-mono">{v}</span>
            </div>
          ))}
        </GlassCard>
      </div>
    </div>
  );
}
