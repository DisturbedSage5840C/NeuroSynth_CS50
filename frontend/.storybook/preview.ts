// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { Preview } from "@storybook/react";
import "../src/index.css";

const preview: Preview = {
  parameters: {
    layout: "fullscreen",
    backgrounds: {
      default: "clinical",
      values: [{ name: "clinical", value: "#08121a" }],
    },
  },
};

export default preview;
