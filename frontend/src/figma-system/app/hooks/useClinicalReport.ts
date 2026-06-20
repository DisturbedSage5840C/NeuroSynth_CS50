// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useMutation } from '@tanstack/react-query';
import { apiFetch } from '../../../lib/api';

export function useGenerateReport() {
  return useMutation({
    mutationFn: ({ patient_id, notes }: { patient_id: string; notes?: string }) =>
      apiFetch('/reports/generate', {
        method: 'POST',
        body: JSON.stringify({ patient_id, notes }),
      }),
  });
}
