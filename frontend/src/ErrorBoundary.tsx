import { Component, type ReactNode } from "react";

export default class ErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed)
      return (
        <main className="recovery-screen">
          <h1>The workspace could not be displayed</h1>
          <p>
            The Python engine is separate from this window. Reload the interface
            to reconnect.
          </p>
          <button className="button" onClick={() => location.reload()}>
            Reload workspace
          </button>
        </main>
      );
    return this.props.children;
  }
}
