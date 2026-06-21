// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { Meta, StoryObj } from "@storybook/react";
import { PanelCard } from "./PanelCard";

const meta: Meta<typeof PanelCard> = {
  title: "Design/PanelCard",
  component: PanelCard,
};

export default meta;
type Story = StoryObj<typeof PanelCard>;

export const Default: Story = {
  args: {
    title: "Clinical Summary",
    subtitle: "Patient-level interpretation",
    children: <p className="text-sm text-muted">This panel hosts one major clinical insight block.</p>,
  },
};
