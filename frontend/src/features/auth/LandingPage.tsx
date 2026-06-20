// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useRef, useMemo } from 'react';
import { Link } from 'react-router';
import { Canvas, useFrame } from '@react-three/fiber';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { ArrowRight, Brain, Activity, FlaskConical, Dna } from 'lucide-react';
import './landing.css';

// ── Neural Network 3D Scene ──────────────────────────────────────────────────

const NODE_COUNT = 52;
const CYAN = new THREE.Color('#00d4ff');
const CYAN_DIM = new THREE.Color('#007a99');

function NeuralNodes({ positions }: { positions: THREE.Vector3[] }) {
  const groupRef = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (!groupRef.current) return;
    const t = clock.getElapsedTime();
    groupRef.current.rotation.y = t * 0.08;
    groupRef.current.rotation.x = Math.sin(t * 0.05) * 0.15;
  });
  return (
    <group ref={groupRef}>
      {positions.map((pos, i) => (
        <mesh key={i} position={pos}>
          <sphereGeometry args={[0.045, 8, 8]} />
          <meshBasicMaterial color={CYAN} transparent opacity={0.75} />
        </mesh>
      ))}
    </group>
  );
}

function NeuralEdges({ positions }: { positions: THREE.Vector3[] }) {
  const groupRef = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (!groupRef.current) return;
    const t = clock.getElapsedTime();
    groupRef.current.rotation.y = t * 0.08;
    groupRef.current.rotation.x = Math.sin(t * 0.05) * 0.15;
  });
  const geometry = useMemo(() => {
    const verts: number[] = [];
    for (let i = 0; i < positions.length; i++) {
      positions
        .map((p, j) => ({ j, d: positions[i].distanceTo(p) }))
        .sort((a, b) => a.d - b.d)
        .slice(1, 4)
        .forEach(({ j }) => {
          if (i < j) {
            verts.push(positions[i].x, positions[i].y, positions[i].z);
            verts.push(positions[j].x, positions[j].y, positions[j].z);
          }
        });
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    return geo;
  }, [positions]);
  return (
    <group ref={groupRef}>
      <lineSegments geometry={geometry}>
        <lineBasicMaterial color={CYAN_DIM} transparent opacity={0.12} />
      </lineSegments>
    </group>
  );
}

function NeuralScene() {
  const positions = useMemo(() => {
    let s = 42;
    const rand = () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
    return Array.from({ length: NODE_COUNT }, () => {
      const theta = rand() * Math.PI * 2;
      const phi = Math.acos(2 * rand() - 1);
      const r = 1.9 + rand() * 0.5;
      return new THREE.Vector3(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi),
      );
    });
  }, []);
  return (
    <>
      <ambientLight intensity={0.2} />
      <NeuralNodes positions={positions} />
      <NeuralEdges positions={positions} />
    </>
  );
}

// ── Static data ───────────────────────────────────────────────────────────────

const METRICS = [
  { value: '0.994', label: 'AUC',      sub: 'real patient data' },
  { value: '6',     label: 'Diseases', sub: 'AD · PD · MS · EP · ALS · HD' },
  { value: '56',    label: 'Features', sub: 'clinical + genomic + wearable' },
  { value: '48 mo', label: 'Forecast', sub: 'trajectory horizon' },
];

const FEATURES = [
  { Icon: Brain,        title: 'CatBoost Ensemble',  desc: '6-model stack with conformal prediction intervals' },
  { Icon: Activity,     title: '48-Month Trajectory', desc: 'TFT with disease-specific monotone constraints' },
  { Icon: FlaskConical, title: 'RAG SOAP Reports',    desc: 'PubMed-grounded SOAP notes with inline citations' },
  { Icon: Dna,          title: 'Genomic Transformer', desc: 'Hierarchical variant encoder with MC Dropout' },
];

// ── Component ─────────────────────────────────────────────────────────────────

export function LandingPage() {
  return (
    <div className="lp-root">
      {/* Three.js canvas — full-screen background */}
      <div className="lp-canvas-wrap" aria-hidden>
        <Canvas camera={{ position: [0, 0, 5.5], fov: 55 }} gl={{ antialias: true, alpha: true }}>
          <NeuralScene />
        </Canvas>
        <div className="lp-canvas-overlay" />
      </div>

      <div className="lp-content">
        {/* Nav */}
        <header className="lp-nav">
          <div className="lp-logo">
            <div className="lp-logo-icon">
              <Brain size={16} />
            </div>
            <span className="lp-logo-name">NeuroSynth</span>
          </div>
          <Link to="/login" className="lp-signin">Sign in</Link>
        </header>

        {/* Hero */}
        <main className="lp-hero">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          >
            <div className="lp-badge">v5 · REAL PATIENT DATA · AUC 0.994</div>
            <h1 className="lp-h1">
              Clinical AI for{' '}
              <span className="lp-h1-accent">Neurological</span>
              <br />Risk Prediction
            </h1>
            <p className="lp-sub">
              NeuroSynth predicts Alzheimer's, Parkinson's, MS, Epilepsy, ALS, and
              Huntington's from a single clinical profile — with causal explanations,
              48-month trajectories, and LLM-generated SOAP reports.
            </p>
            <div className="lp-ctas">
              <Link to="/login" className="lp-cta-primary">
                Enter Clinical Portal <ArrowRight size={15} />
              </Link>
              <Link to="/login" className="lp-cta-secondary">View Demo</Link>
            </div>
          </motion.div>

          {/* Metrics strip */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35, duration: 0.5 }}
            className="lp-metrics"
          >
            {METRICS.map((m) => (
              <div key={m.label} className="lp-metric">
                <div className="lp-metric-value">{m.value}</div>
                <div className="lp-metric-label">{m.label}</div>
                <div className="lp-metric-sub">{m.sub}</div>
              </div>
            ))}
          </motion.div>
        </main>

        {/* Feature grid */}
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="lp-features"
        >
          <div className="lp-feature-grid">
            {FEATURES.map(({ Icon, title, desc }, i) => (
              <motion.div
                key={title}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + i * 0.07 }}
                className="lp-feature-card"
              >
                <div className="lp-feature-icon"><Icon size={16} /></div>
                <div className="lp-feature-title">{title}</div>
                <div className="lp-feature-desc">{desc}</div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        <footer className="lp-footer">
          v5.0 · 16,000+ real patient records · Research use only · Not a diagnostic device
        </footer>
      </div>
    </div>
  );
}
