// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { Meta, StoryObj } from "@storybook/react";
import { ConfidenceBadge } from "./ConfidenceBadge";

const meta: Meta<typeof ConfidenceBadge> = {
  title: "Design/ConfidenceBadge",
  component: ConfidenceBadge,
};

export default meta;
type Story = StoryObj<typeof ConfidenceBadge>;

export const High: Story = { args: { confidence: 0.91 } };
export const Medium: Story = { args: { confidence: 0.66 } };
export const Low: Story = { args: { confidence: 0.24 } };
