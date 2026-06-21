// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { Meta, StoryObj } from "@storybook/react";
import { BrainConnectomeForceGraph } from "./BrainConnectomeForceGraph";

const meta: Meta<typeof BrainConnectomeForceGraph> = {
  title: "Panels/BrainConnectomeForceGraph",
  component: BrainConnectomeForceGraph,
};

export default meta;
type Story = StoryObj<typeof BrainConnectomeForceGraph>;

export const Default: Story = {
  args: {
    nodes: [
      { id: "Hippocampus", risk: 0.82, structuralMetric: 0.34 },
      { id: "Amygdala", risk: 0.63, structuralMetric: 0.47 },
      { id: "Entorhinal", risk: 0.74, structuralMetric: 0.39 },
      { id: "Thalamus", risk: 0.45, structuralMetric: 0.52 },
    ],
    edges: [
      { from: "Hippocampus", to: "Amygdala", strength: 0.65 },
      { from: "Hippocampus", to: "Entorhinal", strength: 0.71 },
      { from: "Entorhinal", to: "Thalamus", strength: 0.44 },
    ],
  },
};
