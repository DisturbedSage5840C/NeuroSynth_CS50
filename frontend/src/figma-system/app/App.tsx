// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { Component, type ReactNode } from 'react';
import { RouterProvider } from 'react-router';
import { router } from './routes';

interface ErrorBoundaryState { error: Error | null }

class RootErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 40, fontFamily: 'monospace', background: '#0a0a0a', color: '#ff4444', minHeight: '100vh' }}>
          <h2 style={{ color: '#ff4444' }}>App crashed — check this error:</h2>
          <pre style={{ whiteSpace: 'pre-wrap', color: '#ffaaaa', marginTop: 16 }}>
            {this.state.error.message}
            {'\n\n'}
            {this.state.error.stack}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <RootErrorBoundary>
      <RouterProvider router={router} />
    </RootErrorBoundary>
  );
}
