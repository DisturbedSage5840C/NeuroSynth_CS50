// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { motion } from 'framer-motion';
import { Brain, Loader2 } from 'lucide-react';
import { login } from '@/lib/api';
import { useAuthStore } from '@/state/authStore';
import './login.css';

const ROLES = ['CLINICIAN', 'RESEARCHER', 'ADMIN'] as const;
type Role = (typeof ROLES)[number];

export function LoginPage() {
  const navigate    = useNavigate();
  const setTokens   = useAuthStore((s) => s.setTokens);
  const hasRole     = useAuthStore((s) => s.role);

  const [username, setUsername] = useState('clinician@neurosynth.local');
  const [password, setPassword] = useState('neurosynth');
  const [role, setRole]         = useState<Role>('CLINICIAN');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  useEffect(() => {
    if (hasRole || localStorage.getItem('ns_logged_in') === 'true') {
      navigate('/app', { replace: true });
    }
  }, [hasRole, navigate]);

  const onSubmit = async () => {
    setLoading(true);
    setError('');
    try {
      const payload = await login(username, password, role);
      setTokens(payload.access_token, payload.refresh_token, payload.role);
      localStorage.setItem('ns_logged_in', 'true');
      navigate('/app', { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message || 'Login failed' : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') onSubmit();
  };

  return (
    <div className="login-root">
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="login-card"
      >
        {/* Header */}
        <div className="login-header">
          <div className="login-logo-ring">
            <Brain size={20} />
          </div>
          <h1 className="login-title">NeuroSynth</h1>
          <p className="login-subtitle">Clinical Neurological AI Platform</p>
        </div>

        {/* Username */}
        <div className="login-field">
          <label className="login-label" htmlFor="ns-user">Username</label>
          <input
            id="ns-user"
            className="login-input"
            type="email"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={onKey}
          />
        </div>

        {/* Password */}
        <div className="login-field">
          <label className="login-label" htmlFor="ns-pass">Password</label>
          <input
            id="ns-pass"
            className="login-input"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={onKey}
          />
        </div>

        {/* Role */}
        <div className="login-field">
          <label className="login-label">Role</label>
          <div className="login-roles">
            {ROLES.map((r) => (
              <button
                key={r}
                type="button"
                className={`login-role-btn${role === r ? ' active' : ''}`}
                onClick={() => setRole(r)}
              >
                {r}
              </button>
            ))}
          </div>
        </div>

        {/* Submit */}
        <button
          type="button"
          className="login-submit"
          onClick={onSubmit}
          disabled={loading}
        >
          {loading ? <Loader2 size={16} className="animate-spin" /> : null}
          {loading ? 'Signing in…' : 'Enter Clinical Portal'}
        </button>

        {error && <div className="login-error">{error}</div>}

        <Link to="/" className="login-back">← Back to home</Link>
      </motion.div>
    </div>
  );
}
