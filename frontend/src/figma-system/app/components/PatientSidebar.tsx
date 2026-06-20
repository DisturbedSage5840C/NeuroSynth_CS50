// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useEffect, useRef, useState } from 'react';
import { Search, ChevronDown, ChevronUp, Activity, Brain, User } from 'lucide-react';
import { patients as mockPatients, type Patient } from '../data/mock-data';
import { usePatients } from '../hooks/usePatients';
import { RiskBadge } from './UncertaintyBadge';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../../../lib/api';
import { useAnalysisStore } from '../../../state/analysisStore';
import { useAuthStore } from '../../../state/authStore';

interface PatientSidebarProps {
  selectedId: string;
  onSelect: (id: string) => void;
}

type SortKey = 'deteriorationProb' | 'name' | 'lastUpdated';

export function PatientSidebar({ selectedId, onSelect }: PatientSidebarProps) {
  const { data: patients = [], isLoading } = usePatients();
  const analysisResult = useAnalysisStore((s) => s.result);
  const accessToken = useAuthStore((s) => s.accessToken);
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('deteriorationProb');
  const [sortAsc, setSortAsc] = useState(false);
  const lastAutoSelected = useRef<string | null>(null);

  // Only fall back to mock patients when not logged in (demo mode).
  // When logged in, show real patients (or empty list while loading).
  const displayPatients: Patient[] = accessToken
    ? patients
    : (patients.length ? patients : mockPatients);

  const filtered = displayPatients
    .filter(p => p.name.toLowerCase().includes(search.toLowerCase()) || p.mrn.includes(search))
    .sort((a, b) => {
      const mul = sortAsc ? 1 : -1;
      if (sortKey === 'name') return mul * a.name.localeCompare(b.name);
      if (sortKey === 'deteriorationProb') return mul * (a.deteriorationProb - b.deteriorationProb);
      return 0;
    });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(false); }
  };

  useEffect(() => {
    if (!filtered.length) return;
    const hasSelected = filtered.some((p) => p.id === selectedId);
    const nextId = filtered[0].id;
    if (!hasSelected && lastAutoSelected.current !== nextId) {
      lastAutoSelected.current = nextId;
      onSelect(nextId);
    }
  }, [filtered, selectedId, onSelect]);

  const createPatient = useMutation({
    mutationFn: () => {
      const suffix = String(Date.now()).slice(-4);
      const params = new URLSearchParams({
        name: `New Patient ${suffix}`,
        age: '62',
        sex: 'F',
        diagnosis: 'Neurology Monitoring',
      });
      return apiFetch<{ patient_id: string }>(`/patients/?${params.toString()}`, {
        method: 'POST',
      });
    },
    onSuccess: async (data) => {
      await queryClient.invalidateQueries({ queryKey: ['patients'] });
      if (data?.patient_id) onSelect(data.patient_id);
    },
  });

  if (isLoading) return <SidebarSkeleton />;

  return (
    <div className="w-72 h-full flex flex-col border-r border-border bg-[var(--sidebar)]">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2 mb-3">
          <Brain size={18} className="text-primary" />
          <span className="font-mono tracking-wider" style={{ fontSize: '11px', color: 'var(--muted-foreground)' }}>NEUROSYNTH</span>
        </div>
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search patients or MRN..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 rounded-md bg-[var(--input-background)] border border-border text-foreground placeholder:text-muted-foreground focus:border-primary focus:ring-1 focus:ring-primary/30 transition-colors"
            style={{ fontSize: '12px' }}
          />
        </div>
        <button
          onClick={() => createPatient.mutate()}
          disabled={createPatient.isPending}
          className="mt-3 w-full rounded-md border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground disabled:opacity-50"
        >
          {createPatient.isPending ? 'Creating...' : 'Add New Patient'}
        </button>
      </div>

      {/* Sort controls */}
      <div className="px-4 py-2 flex items-center gap-2 border-b border-border">
        <span className="text-muted-foreground" style={{ fontSize: '10px', letterSpacing: '0.05em' }}>SORT BY</span>
        <button
          onClick={() => handleSort('deteriorationProb')}
          className={`px-2 py-0.5 rounded text-[10px] transition-colors ${sortKey === 'deteriorationProb' ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
        >
          RISK {sortKey === 'deteriorationProb' && (sortAsc ? <ChevronUp size={10} className="inline" /> : <ChevronDown size={10} className="inline" />)}
        </button>
        <button
          onClick={() => handleSort('name')}
          className={`px-2 py-0.5 rounded text-[10px] transition-colors ${sortKey === 'name' ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
        >
          NAME {sortKey === 'name' && (sortAsc ? <ChevronUp size={10} className="inline" /> : <ChevronDown size={10} className="inline" />)}
        </button>
      </div>

      {/* Patient list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.map(patient => (
          (() => {
            const isCurrentlyAnalyzed = analysisResult?.patient_id === patient.id;
            const displayProb = isCurrentlyAnalyzed
              ? analysisResult!.probability
              : patient.deteriorationProb;
            const displayRisk = isCurrentlyAnalyzed
              ? (analysisResult!.risk_level.toLowerCase().includes('critical') ? 'critical' :
                 analysisResult!.risk_level.toLowerCase().includes('high') ? 'high' :
                 analysisResult!.risk_level.toLowerCase().includes('moderate') ? 'moderate' : 'low')
              : patient.riskLevel;

            return (
          <button
            key={patient.id}
            onClick={() => onSelect(patient.id)}
            className={`w-full text-left px-4 py-3 border-b border-border transition-colors ${
              selectedId === patient.id
                ? 'bg-primary/8 border-l-2 border-l-primary'
                : 'hover:bg-[var(--sidebar-accent)] border-l-2 border-l-transparent'
            }`}
          >
            <div className="flex items-start justify-between mb-1">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-full bg-secondary flex items-center justify-center">
                  <User size={12} className="text-muted-foreground" />
                </div>
                <div>
                  <div className="text-foreground" style={{ fontSize: '12px' }}>{patient.name}</div>
                  <div className="text-muted-foreground font-mono" style={{ fontSize: '10px' }}>{patient.mrn} · {patient.age}{patient.sex}</div>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between mt-1.5 pl-8">
              <RiskBadge level={displayRisk} value={`${Math.round(displayProb * 100)}%`} />
              <span className="text-muted-foreground" style={{ fontSize: '10px' }}>
                <Activity size={10} className="inline mr-1" />
                {patient.lastUpdated}
              </span>
            </div>
            {isCurrentlyAnalyzed && (
              <div className="text-[10px] text-primary mt-1 pl-8">Last analyzed just now</div>
            )}
            <div className="text-muted-foreground mt-1 pl-8" style={{ fontSize: '10px' }}>{patient.diagnosis}</div>
          </button>
            );
          })()
        ))}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-border">
        <div className="flex items-center justify-between text-muted-foreground" style={{ fontSize: '10px' }}>
          <span>{filtered.length} patients</span>
          <span className="font-mono">v3.2.1</span>
        </div>
      </div>
    </div>
  );
}

function SidebarSkeleton() {
  return (
    <div className="w-72 h-full flex flex-col border-r border-border bg-[var(--sidebar)] p-4 gap-3">
      <div className="h-4 w-28 rounded bg-secondary animate-pulse" />
      <div className="h-8 w-full rounded bg-secondary animate-pulse" />
      <div className="space-y-2 mt-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-14 w-full rounded bg-secondary animate-pulse" />
        ))}
      </div>
    </div>
  );
}
