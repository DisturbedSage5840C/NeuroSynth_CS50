// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface TFTForecastPoint {
  month: number;
  mean: number;
  lower80: number;
  upper80: number;
  lower95: number;
  upper95: number;
}

export interface TFTForecastChartProps {
  data: TFTForecastPoint[];
}

export function TFTForecastChart({ data }: TFTForecastChartProps): JSX.Element {
  return (
    <div className="h-80 w-full" aria-label="12-month deterioration forecast chart">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 12, right: 16, left: 4, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(120,140,160,0.3)" />
          <XAxis dataKey="month" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Area type="monotone" dataKey="upper95" stroke="none" fill="rgba(14,165,233,0.12)" />
          <Area type="monotone" dataKey="lower95" stroke="none" fill="hsl(var(--surface))" />
          <Area type="monotone" dataKey="upper80" stroke="none" fill="rgba(45,212,191,0.18)" />
          <Area type="monotone" dataKey="lower80" stroke="none" fill="hsl(var(--surface))" />
          <Line type="monotone" dataKey="mean" stroke="#22d3ee" strokeWidth={3} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
