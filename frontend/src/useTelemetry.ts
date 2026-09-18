import {useEffect, useState} from 'react';
import {validSnapshot, type Snapshot} from './types';

export function useTelemetry() {
  const [state, setState] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    let socket: WebSocket; let retry: ReturnType<typeof setTimeout>; let disposed = false; let last = 0;
    const connect = () => {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/v1/telemetry`);
      socket.onmessage = e => {
        try {
          const value: unknown = JSON.parse(e.data);
          if (!validSnapshot(value)) throw new Error('Unsupported telemetry format');
          last = Date.now(); setState(value); setConnected(true); setError('');
        } catch { setError('The server sent an unsupported state. Restart with matching app versions.'); setConnected(false); }
      };
      socket.onclose = () => {setConnected(false); if (!disposed) retry = setTimeout(connect, 1500);};
      socket.onerror = () => socket.close();
    };
    connect();
    const watchdog = setInterval(() => {if (Date.now() - last > 2500) setConnected(false);}, 500);
    return () => {disposed = true; clearTimeout(retry); clearInterval(watchdog); socket.close();};
  }, []);
  return {state, connected: connected && (state?.age_ms ?? Infinity) < 2000, error};
}
