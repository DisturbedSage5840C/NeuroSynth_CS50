// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { Meta, StoryObj } from "@storybook/react";
import { ClinicalReportRenderer } from "./ClinicalReportRenderer";

const meta: Meta<typeof ClinicalReportRenderer> = {
  title: "Panels/ClinicalReportRenderer",
  component: ClinicalReportRenderer,
};

export default meta;
type Story = StoryObj<typeof ClinicalReportRenderer>;

export const Default: Story = {
  args: {
    report: {
      assessment: "Patient shows moderate short-term deterioration risk driven by cognitive and behavioral markers.",
      risk_level: "moderate",
      uncertainty_note: "Prediction intervals widen after month 8, requiring clinician review and repeated labs.",
      evidence_refs: ["EVID-001", "EVID-004", "EVID-017"],
      causal_pathways: [
        { pathway: "SleepQuality -> MMSE -> Risk", rationale: "Sleep deficits accelerate cognitive decline.", confidence: 0.74 },
        { pathway: "Inflammation -> NfL -> FunctionalAssessment", rationale: "Neuroinflammation elevates injury markers.", confidence: 0.68 },
      ],
      interventions: [
        { name: "Sleep stabilization", action: "Cognitive behavioral sleep protocol", priority: 1, expected_impact: "May reduce medium-term risk by 8-12%", confidence: 0.71 },
        { name: "Guided physical activity", action: "3 sessions weekly with progression", priority: 2, expected_impact: "May slow functional decline", confidence: 0.64 },
      ],
    },
  },
};
