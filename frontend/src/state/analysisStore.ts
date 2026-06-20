// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { create } from "zustand";
import type { AnalysisResult } from "../figma-system/app/types/analysis";

interface AnalysisState {
  result: AnalysisResult | null;
  setResult: (result: AnalysisResult | null) => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  result: null,
  setResult: (result) => set({ result }),
}));
