// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { Meta, StoryObj } from "@storybook/react";
import { GenomicRiskHeatmap } from "./GenomicRiskHeatmap";

const meta: Meta<typeof GenomicRiskHeatmap> = {
  title: "Panels/GenomicRiskHeatmap",
  component: GenomicRiskHeatmap,
};

export default meta;
type Story = StoryObj<typeof GenomicRiskHeatmap>;

export const Default: Story = {
  args: {
    data: ["chr1", "chr2", "chr11", "chr17", "chr19"].flatMap((chrom, i) =>
      ["p-arm", "q-arm", "band-1", "band-2", "band-3"].map((region, j) => ({
        chromosome: chrom,
        region,
        risk: Math.min(1, 0.1 + (i + j) * 0.09),
      }))
    ),
  },
};
