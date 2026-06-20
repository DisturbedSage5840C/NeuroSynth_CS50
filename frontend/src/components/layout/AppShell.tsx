// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { PropsWithChildren } from "react";
import { Link } from "react-router-dom";

export function AppShell({ children }: PropsWithChildren): JSX.Element {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-line bg-surface/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 tablet:px-8">
          <h1 className="text-xl font-bold text-accent">NeuroSynth Clinical Console</h1>
          <nav className="flex gap-4 text-sm text-muted">
            <Link to="/">Dashboard</Link>
            <Link to="/reports">Reports</Link>
            <Link to="/login">Login</Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto grid max-w-7xl gap-4 px-4 py-4 tablet:px-8">{children}</main>
    </div>
  );
}
