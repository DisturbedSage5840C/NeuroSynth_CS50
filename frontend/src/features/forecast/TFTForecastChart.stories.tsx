// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { Meta, StoryObj } from "@storybook/react";
import { TFTForecastChart } from "./TFTForecastChart";

const meta: Meta<typeof TFTForecastChart> = {
  title: "Panels/TFTForecastChart",
  component: TFTForecastChart,
};

export default meta;
type Story = StoryObj<typeof TFTForecastChart>;

export const Default: Story = {
  args: {
    data: Array.from({ length: 12 }).map((_, i) => {
      const mean = Math.min(0.92, 0.22 + i * 0.05);
      const w95 = 0.08 + i * 0.015;
      const w80 = w95 * 0.6;
      return {
        month: i + 1,
        mean,
        lower80: Math.max(0, mean - w80),
        upper80: Math.min(1, mean + w80),
        lower95: Math.max(0, mean - w95),
        upper95: Math.min(1, mean + w95),
      };
    }),
  },
};
