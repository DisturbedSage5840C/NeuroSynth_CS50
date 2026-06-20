// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { Group } from "@visx/group";
import { scaleBand, scaleLinear } from "@visx/scale";
import { HeatmapRect } from "@visx/heatmap";

export interface HeatmapCell {
  chromosome: string;
  region: string;
  risk: number;
}

export interface GenomicRiskHeatmapProps {
  data: HeatmapCell[];
  width?: number;
  height?: number;
}

const color = scaleLinear<string>({ domain: [0, 0.5, 1], range: ["#14b8a6", "#fb923c", "#ef4444"] });

export function GenomicRiskHeatmap({ data, width = 760, height = 320 }: GenomicRiskHeatmapProps): JSX.Element {
  const chromosomes = [...new Set(data.map((d) => d.chromosome))];
  const regions = [...new Set(data.map((d) => d.region))];

  const xScale = scaleBand<string>({ domain: chromosomes, range: [0, width], padding: 0.08 });
  const yScale = scaleBand<string>({ domain: regions, range: [0, height], padding: 0.08 });

  const grouped = chromosomes.map((chromosome) => ({
    bin: chromosome,
    bins: regions.map((region) => {
      const match = data.find((d) => d.chromosome === chromosome && d.region === region);
      return { bin: region, count: match?.risk ?? 0 };
    }),
  }));

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height + 30}`} aria-label="Genomic risk heatmap" className="rounded-panel border border-line bg-surface">
      <Group left={0} top={0}>
        <HeatmapRect
          data={grouped}
          xScale={(d) => xScale(d) ?? 0}
          yScale={(d) => yScale(d) ?? 0}
          colorScale={(v) => color(v) || "#14b8a6"}
          binWidth={xScale.bandwidth()}
          binHeight={yScale.bandwidth()}
          gap={2}
        >
          {(heatmap) =>
            heatmap.map((bins) =>
              bins.map((bin) => (
                <rect
                  key={`${bin.row}-${bin.column}`}
                  x={bin.x}
                  y={bin.y}
                  width={bin.width}
                  height={bin.height}
                  fill={bin.color}
                  rx={4}
                />
              ))
            )
          }
        </HeatmapRect>
      </Group>
    </svg>
  );
}
