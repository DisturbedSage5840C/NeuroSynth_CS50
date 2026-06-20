// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
/**
 * BrainVisualization3D v2 — 116-region AAL atlas with SHAP coloring.
 *
 * Regions from the AAL-116 atlas (Tzourio-Mazoyer et al. 2002) positioned using
 * approximate MNI centroids. Each region is a noise-displaced icosphere colored
 * by its aggregated SHAP contribution. Click a region to see mapped features.
 * Disease-focus mode dims all regions not associated with the selected disease.
 *
 * Lazy-load with React.lazy — Three.js stays out of the main bundle.
 */
import { Canvas } from '@react-three/fiber';
import { Html, OrbitControls } from '@react-three/drei';
import { useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { AAL_REGIONS, mapFeatureToAAL, type AALRegion } from '@/lib/aalAtlas';

interface SHAPValue { feature: string; value: number; }

// ── Color mapping ─────────────────────────────────────────────────────────────

function colorForShap(v: number, dimmed: boolean): THREE.Color {
  if (dimmed) return new THREE.Color('#1c2a42');
  if (v > 0.05) {
    const t = Math.min(v / 0.3, 1);
    return new THREE.Color().lerpColors(new THREE.Color('#64748b'), new THREE.Color('#ef4444'), 0.4 + 0.6 * t);
  }
  if (v < -0.05) {
    const t = Math.min(-v / 0.3, 1);
    return new THREE.Color().lerpColors(new THREE.Color('#64748b'), new THREE.Color('#22c55e'), 0.4 + 0.6 * t);
  }
  return new THREE.Color('#2a3a52');
}

// ── Aggregate SHAP values onto AAL regions ────────────────────────────────────

function buildRegionMap(shapValues: SHAPValue[]): Map<string, number> {
  const agg = new Map<string, number>();
  for (const sv of shapValues) {
    const regionId = mapFeatureToAAL(sv.feature);
    agg.set(regionId, (agg.get(regionId) ?? 0) + sv.value);
  }
  return agg;
}

function featuresForRegion(regionId: string, shapValues: SHAPValue[]): string[] {
  return shapValues
    .filter((sv) => mapFeatureToAAL(sv.feature) === regionId)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 3)
    .map((sv) => sv.feature);
}

// ── Region mesh ───────────────────────────────────────────────────────────────

function RegionMesh({
  region, shapValue, dimmed, selected, onSelect,
}: {
  region: AALRegion;
  shapValue: number;
  dimmed: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const color = useMemo(() => colorForShap(shapValue, dimmed), [shapValue, dimmed]);

  const geometry = useMemo(() => {
    const geo = new THREE.IcosahedronGeometry(region.scale, 2);
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      // Light sulci displacement — less noise for smaller regions
      const noise = Math.sin(v.x * 11) * Math.cos(v.y * 11) * Math.sin(v.z * 11);
      v.multiplyScalar(1 + noise * 0.04);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
    return geo;
  }, [region.scale]);

  const emissiveIntensity = selected ? 0.6 : hovered ? 0.4 : dimmed ? 0.0 : 0.1;
  const opacity = dimmed ? 0.25 : selected ? 1.0 : hovered ? 0.95 : 0.85;

  return (
    <mesh
      geometry={geometry}
      position={region.position}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true); }}
      onPointerOut={() => setHovered(false)}
      onClick={(e) => { e.stopPropagation(); onSelect(); }}
    >
      <meshStandardMaterial
        color={color}
        roughness={0.6}
        metalness={0.05}
        transparent
        opacity={opacity}
        emissive={color}
        emissiveIntensity={emissiveIntensity}
      />
      {(hovered || selected) && (
        <Html
          distanceFactor={7}
          position={[0, region.scale + 0.06, 0]}
          center
        >
          <div className="brain-tooltip">
            <span className="brain-tooltip-name">{region.name}</span>
            {shapValue !== 0 && (
              <span className={`brain-tooltip-val ${shapValue > 0 ? 'brain-tooltip-pos' : 'brain-tooltip-neg'}`}>
                {shapValue > 0 ? '+' : ''}{shapValue.toFixed(3)}
              </span>
            )}
          </div>
        </Html>
      )}
    </mesh>
  );
}

// ── Public helper — maps SHAP values to { regionId, shap } objects ───────────

export interface BrainRegion {
  id: string;
  label: string;
  shap: number;
}

export function regionsFromShap(shapValues: SHAPValue[]): BrainRegion[] {
  const agg = buildRegionMap(shapValues);
  return AAL_REGIONS.map((r) => ({
    id: r.id,
    label: r.label,
    shap: agg.get(r.id) ?? 0,
  }));
}

// ── Main component ────────────────────────────────────────────────────────────

interface BrainVisualization3DProps {
  shapValues?: SHAPValue[];
  focusDisease?: string | null;
  height?: number;
}

export default function BrainVisualization3D({
  shapValues = [],
  focusDisease = null,
  height = 400,
}: BrainVisualization3DProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const shapMap = useMemo(() => buildRegionMap(shapValues), [shapValues]);

  const selectedFeatures = useMemo(() => {
    if (!selectedId) return [];
    return featuresForRegion(selectedId, shapValues);
  }, [selectedId, shapValues]);

  const selectedRegion = AAL_REGIONS.find((r) => r.id === selectedId);

  return (
    <div className="brain-root" style={{ height }}>
      <div className="brain-canvas-wrap">
        <Canvas camera={{ position: [0, 0.5, 4.2], fov: 42 }} dpr={[1, 2]}>
          <ambientLight intensity={0.45} />
          <directionalLight position={[4, 5, 5]} intensity={0.75} />
          <directionalLight position={[-4, -2, -3]} intensity={0.25} color="#00d4aa" />
          {AAL_REGIONS.map((region) => {
            const sv = shapMap.get(region.id) ?? 0;
            const dimmed = focusDisease != null
              ? !(region.diseases ?? []).includes(focusDisease)
              : false;
            return (
              <RegionMesh
                key={region.id}
                region={region}
                shapValue={sv}
                dimmed={dimmed}
                selected={selectedId === region.id}
                onSelect={() => setSelectedId((prev) => prev === region.id ? null : region.id)}
              />
            );
          })}
          <OrbitControls
            enableZoom
            enablePan={false}
            autoRotate={!selectedId}
            autoRotateSpeed={0.5}
            minDistance={2.5}
            maxDistance={7}
          />
        </Canvas>
      </div>

      {/* Selected region panel */}
      {selectedRegion && (
        <div className="brain-detail">
          <div className="brain-detail-name">{selectedRegion.name}</div>
          <div className="brain-detail-lobe">{selectedRegion.lobe}</div>
          {selectedFeatures.length > 0 && (
            <div className="brain-detail-features">
              {selectedFeatures.map((f) => (
                <span key={f} className="brain-detail-feature-chip">{f}</span>
              ))}
            </div>
          )}
          <button
            type="button"
            className="brain-detail-close"
            onClick={() => setSelectedId(null)}
          >
            ×
          </button>
        </div>
      )}

      <div className="brain-count">{AAL_REGIONS.length} regions · AAL-116</div>
    </div>
  );
}
