// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useEffect, useRef, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { streamUrl } from '../../../lib/api';
import type { BiomarkerReading } from '../data/mock-data';
import { biomarkerHistory } from '../data/mock-data';

const MAX_POINTS = 60;

interface BiomarkerStripProps {
  patientId?: string;
}

export function BiomarkerStrip({ patientId = 'P-001' }: BiomarkerStripProps) {
  const [readings, setReadings] = useState<BiomarkerReading[]>(biomarkerHistory.slice(-20));
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const subscribe = (url: string) => {
      const es = new EventSource(url, { withCredentials: true });
      esRef.current = es;
      es.onmessage = (event) => {
        const data: BiomarkerReading = JSON.parse(event.data);
        setReadings((prev) => [...prev.slice(-MAX_POINTS + 1), data]);
      };
      return es;
    };

    try {
      const primary = subscribe(streamUrl(`/biomarkers/live/${patientId}`));
      primary.onerror = () => {
        primary.close();
        try {
          const fallback = subscribe(streamUrl('/biomarkers/stream'));
          fallback.onerror = () => fallback.close();
        } catch {
          primary.close();
        }
      };
    } catch {
      let i = 0;
      const interval = setInterval(() => {
        const base = biomarkerHistory[i % biomarkerHistory.length];
        const jitter = () => (Math.random() - 0.5) * 2;
        setReadings((prev) => [...prev.slice(-MAX_POINTS + 1), {
          ...base,
          time: new Date().toLocaleTimeString(),
          heartRate: Math.round(base.heartRate + jitter() * 3),
          spo2: +(base.spo2 + jitter() * 0.3).toFixed(1),
        }]);
        i++;
      }, 2000);
      return () => clearInterval(interval);
    }
    return () => esRef.current?.close();
  }, [patientId]);

  const vitals = [
    { key: 'heartRate', label: 'HR', unit: 'bpm', color: 'var(--risk-critical)', normal: [60, 100] },
    { key: 'spo2', label: 'SpO₂', unit: '%', color: 'var(--chart-2)', normal: [95, 100] },
    { key: 'systolicBP', label: 'SBP', unit: 'mmHg', color: 'var(--chart-3)', normal: [90, 140] },
    { key: 'respiratoryRate', label: 'RR', unit: '/min', color: 'var(--chart-5)', normal: [12, 20] },
  ] as const;

  const latest = readings[readings.length - 1];

  const fmt = (val: number) => Number.isInteger(val) ? String(val) : val.toFixed(1);

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">Real-time Wearable Biomarkers</h3>
        <span className="flex items-center gap-1 text-xs text-[var(--risk-low)]">
          <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
          LIVE
        </span>
      </div>
      <div className="grid grid-cols-4 gap-3">
        {vitals.map(({ key, label, unit, color, normal }) => {
          const val = latest?.[key] ?? 0;
          const isAbnormal = val < normal[0] || val > normal[1];
          return (
            <div key={key} className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span className="text-xs text-muted-foreground">{label}</span>
                <span
                  className="font-mono text-sm font-medium"
                  style={{ color: isAbnormal ? 'var(--risk-critical)' : color }}
                >
                  {fmt(val)}<span className="text-xs text-muted-foreground ml-0.5">{unit}</span>
                </span>
              </div>
              <ResponsiveContainer width="100%" height={60}>
                <LineChart data={readings}>
                  <Line
                    type="monotone"
                    dataKey={key}
                    stroke={isAbnormal ? 'var(--risk-critical)' : color}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <YAxis domain={['auto', 'auto']} hide />
                  <XAxis dataKey="time" hide />
                  <Tooltip
                    contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', fontSize: 11 }}
                    labelStyle={{ color: 'var(--muted-foreground)' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </div>
  );
}
