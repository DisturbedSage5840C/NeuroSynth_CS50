// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useEffect, useState } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router';
import {
  LayoutDashboard, FileText, Database, Brain, BarChart2,
  FlaskConical, BookOpen, Network, Settings, LogOut,
} from 'lucide-react';
import { PatientSidebar } from './PatientSidebar';
import { patients } from '../data/mock-data';
import { useAuthStore } from '../../../state/authStore';
import { useAnalysisStore } from '../../../state/analysisStore';
import './layout.css';

const NAV_PRIMARY = [
  { to: '/app',             Icon: LayoutDashboard, label: 'Dashboard',   end: true  },
  { to: '/app/report',      Icon: FileText,        label: 'Report',      end: false },
  { to: '/app/explorer',    Icon: Database,        label: 'Explorer',    end: false },
  { to: '/app/performance', Icon: BarChart2,       label: 'Performance', end: false },
];

const NAV_V5 = [
  { to: '/app/cohort',     Icon: FlaskConical, label: 'Cohort',      end: false },
  { to: '/app/data',       Icon: Network,      label: 'Data',        end: false },
  { to: '/app/literature', Icon: BookOpen,     label: 'Literature',  end: false },
  { to: '/app/brain',      Icon: Brain,        label: 'Brain Atlas', end: false },
];

export function Layout() {
  const [selectedPatientId, setSelectedPatientId] = useState(patients[0].id);
  const latestAnalysis = useAnalysisStore((s) => s.result);
  const navigate = useNavigate();
  const clear    = useAuthStore((s) => s.clear);

  useEffect(() => {
    if (latestAnalysis?.patient_id) setSelectedPatientId(latestAnalysis.patient_id);
  }, [latestAnalysis?.patient_id]);

  const handleLogout = async () => {
    try {
      await fetch(
        `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/auth/logout`,
        { method: 'POST', credentials: 'include' },
      );
    } catch { /* network error is fine */ }
    clear();
    navigate('/login');
  };

  return (
    <div className="h-screen flex bg-background text-foreground overflow-hidden">
      {/* Nav rail */}
      <nav className="nav-rail">
        <div className="nav-logo">
          <div className="nav-logo-icon">
            <Brain size={13} />
          </div>
          <div>
            <div className="nav-logo-name">NeuroSynth</div>
            <div className="nav-logo-sub">v5 · Clinical AI</div>
          </div>
        </div>

        {NAV_PRIMARY.map(({ to, Icon, label, end }) => (
          <NavItem key={to} to={to} icon={<Icon size={14} />} label={label} end={end} />
        ))}

        <div className="nav-divider" />

        <div className="nav-section-label">v5 features</div>
        {NAV_V5.map(({ to, Icon, label, end }) => (
          <NavItem key={to} to={to} icon={<Icon size={14} />} label={label} end={end} />
        ))}

        <div className="nav-bottom">
          <NavItem to="/app/settings" icon={<Settings size={14} />} label="Settings" end={false} />
          <button type="button" className="nav-logout" onClick={handleLogout}>
            <LogOut size={14} />
            <span>Sign out</span>
          </button>
        </div>
      </nav>

      <PatientSidebar selectedId={selectedPatientId} onSelect={setSelectedPatientId} />
      <Outlet context={{ selectedPatientId }} />
    </div>
  );
}

function NavItem({
  to, icon, label, end,
}: {
  to: string;
  icon: React.ReactNode;
  label: string;
  end: boolean;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `nav-item ${isActive ? 'nav-item-active' : 'nav-item-inactive'}`
      }
    >
      {icon}
      <span>{label}</span>
    </NavLink>
  );
}
