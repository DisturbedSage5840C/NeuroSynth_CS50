// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../../../lib/api';
import type { Patient } from '../data/mock-data';
import { useAuthStore } from '../../../state/authStore';

interface ApiPatientSummary {
  patient_id?: string;
  id?: string;
  name?: string;
  updated_at?: string;
  probability?: number;
  risk_level?: string;
  disease_classification?: { predicted_disease?: string };
}

function toUiPatient(item: ApiPatientSummary, index: number): Patient {
  const id = item.patient_id || item.id || `P-${String(index + 1).padStart(3, '0')}`;
  const nowIso = new Date().toISOString();
  return {
    id,
    name: item.name || `Patient ${id}`,
    age: 60 + (index % 20),
    sex: index % 2 === 0 ? 'M' : 'F',
    mrn: id,
    diagnosis: item.disease_classification?.predicted_disease || 'Neurology Monitoring',
    deteriorationProb: typeof item.probability === 'number' ? item.probability : 0.4,
    riskLevel:
      String(item.risk_level || 'moderate').toLowerCase().includes('critical') ? 'critical' :
      String(item.risk_level || 'moderate').toLowerCase().includes('high') ? 'high' :
      String(item.risk_level || 'moderate').toLowerCase().includes('low') ? 'low' :
      'moderate',
    lastUpdated: item.updated_at || nowIso,
    admissionDate: nowIso.slice(0, 10),
    ward: 'Neuro',
    attendingPhysician: 'Dr. NeuroSynth',
  };
}

export function usePatients() {
  const accessToken = useAuthStore((s) => s.accessToken);

  return useQuery({
    queryKey: ['patients'],
    queryFn: () => apiFetch<{ items: ApiPatientSummary[] }>('/patients'),
    select: (data) => data.items.map(toUiPatient),
    enabled: !!accessToken,
    staleTime: 30_000,
    refetchOnWindowFocus: true,
  });
}
