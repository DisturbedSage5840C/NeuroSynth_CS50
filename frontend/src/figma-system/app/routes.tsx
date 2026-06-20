// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import type { ReactNode } from 'react';
import { createBrowserRouter, Navigate } from 'react-router';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { ReportViewer } from './components/ReportViewer';
import { DataExplorer } from './components/DataExplorer';
import { PerformanceDashboard } from './components/PerformanceDashboard';
import { CohortDashboard } from './components/CohortDashboard';
import { DataPipeline } from './components/DataPipeline';
import { LiteratureSearch } from './components/LiteratureSearch';
import { BrainAtlas } from './components/BrainAtlas';
import { Settings } from './components/Settings';
import { LoginPage } from '../../features/auth/LoginPage';
import { LandingPage } from '../../features/auth/LandingPage';
import { useOutletContext } from 'react-router';
import { useAuthStore } from '../../state/authStore';

function RequireAuth({ children }: { children: ReactNode }) {
  const role = useAuthStore((s) => s.role);
  const hasSession =
    typeof window !== 'undefined' && localStorage.getItem('ns_logged_in') === 'true';
  if (!role && !hasSession) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function DashboardPage() {
  const { selectedPatientId } = useOutletContext<{ selectedPatientId: string }>();
  return <Dashboard selectedPatientId={selectedPatientId} />;
}

export const router = createBrowserRouter([
  { path: '/',      Component: LandingPage },
  { path: '/login', Component: LoginPage  },
  {
    path: '/app',
    element: (
      <RequireAuth>
        <Layout />
      </RequireAuth>
    ),
    children: [
      // ── Core routes ──────────────────────────────────────────────
      { index: true,            Component: DashboardPage      },
      { path: 'report',         Component: ReportViewer       },
      { path: 'explorer',       Component: DataExplorer       },
      { path: 'performance',    Component: PerformanceDashboard },
      // ── v5 routes ────────────────────────────────────────────────
      { path: 'cohort',         Component: CohortDashboard    },
      { path: 'data',           Component: DataPipeline       },
      { path: 'literature',     Component: LiteratureSearch   },
      { path: 'brain',          Component: BrainAtlas         },
      { path: 'settings',       Component: Settings           },
    ],
  },
]);
