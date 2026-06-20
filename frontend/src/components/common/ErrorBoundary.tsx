// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import React from "react";

interface ErrorBoundaryProps {
  title: string;
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, message: error.message };
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div className="rounded-panel border border-danger/50 bg-danger/10 p-4 text-sm" role="alert">
          <p className="font-semibold">{this.props.title} failed to render.</p>
          <p className="text-muted">{this.state.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}
