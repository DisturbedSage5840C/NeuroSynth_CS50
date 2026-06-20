// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts";
import { useWearableStream } from "@/hooks/useWearableStream";

export function RealtimeWearableStrip(): JSX.Element {
  const data = useWearableStream();

  return (
    <div className="h-72 w-full" aria-live="polite" aria-label="Live wearable biomarkers">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 16, left: 4, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(120,140,160,0.3)" />
          <XAxis dataKey="timestamp" tickFormatter={(v) => new Date(v).toLocaleTimeString()} minTickGap={24} />
          <YAxis />
          <Tooltip labelFormatter={(value) => new Date(String(value)).toLocaleString()} />
          <Line type="monotone" dataKey="heart_rate" stroke="#22d3ee" dot={false} />
          <Line type="monotone" dataKey="eeg" stroke="#14b8a6" dot={false} />
          <Line type="monotone" dataKey="accel" stroke="#f59e0b" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
