// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useMutation } from '@tanstack/react-query';
import { apiFetch } from '../../../lib/api';

interface PredictionPayload {
  patient_id: string;
  features: Record<string, number>;
}

export function useRunPrediction() {
  return useMutation({
    mutationFn: (payload: PredictionPayload) =>
      apiFetch('/predictions/run', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
  });
}
