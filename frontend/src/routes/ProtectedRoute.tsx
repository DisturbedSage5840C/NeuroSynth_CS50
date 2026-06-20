// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { Navigate } from "react-router-dom";
import { UserRole, useAuthStore } from "@/state/authStore";

interface ProtectedRouteProps {
  allowed: UserRole[];
  children: JSX.Element;
}

export function ProtectedRoute({ allowed, children }: ProtectedRouteProps): JSX.Element {
  const role = useAuthStore((s) => s.role);
  const token = useAuthStore((s) => s.accessToken);

  if (!token || !role) return <Navigate to="/login" replace />;
  if (!allowed.includes(role)) return <Navigate to="/" replace />;
  return children;
}
