// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
/**
 * Procedural AAL-116 atlas — 116 brain region positions approximated from
 * published MNI coordinates (Tzourio-Mazoyer et al. 2002, NeuroImage 15:273-289).
 *
 * Coordinates are in a normalised [-2, 2] space (MNI mm / 50).
 * x: R+ / L-  y: superior+  z: anterior+
 *
 * Used by BrainVisualization3D to render 116 labeled nodes and aggregate SHAP
 * values by anatomical region instead of broad lobes.
 */

export interface AALRegion {
  id: string;
  name: string;
  lobe: string;
  hemisphere: 'L' | 'R' | 'M';
  position: [number, number, number];
  scale: number;
  diseases?: string[];   // diseases most associated with this region
}

// Lobe → disease associations (for disease-focus mode)
export const LOBE_DISEASE: Record<string, string[]> = {
  frontal:      ["Alzheimer's Disease", "ALS"],
  temporal:     ["Alzheimer's Disease", "Epilepsy"],
  parietal:     ["Alzheimer's Disease", "Multiple Sclerosis"],
  occipital:    ["Multiple Sclerosis", "Epilepsy"],
  limbic:       ["Alzheimer's Disease", "Epilepsy"],
  subcortical:  ["Parkinson's Disease", "Huntington's Disease"],
  cerebellum:   ["Parkinson's Disease", "Multiple Sclerosis"],
  brainstem:    ["ALS", "Parkinson's Disease"],
};

function lr(name: string, lobe: string, mniX: number, mniY: number, mniZ: number, scale = 0.14): [AALRegion, AALRegion] {
  const x = mniX / 50;
  const y = mniY / 50;
  const z = mniZ / 50;
  return [
    { id: `${name}_L`, name: `${name} (L)`, lobe, hemisphere: 'L', position: [-Math.abs(x), y, z], scale, diseases: LOBE_DISEASE[lobe] },
    { id: `${name}_R`, name: `${name} (R)`, lobe, hemisphere: 'R', position: [Math.abs(x), y, z],  scale, diseases: LOBE_DISEASE[lobe] },
  ];
}

function mid(id: string, name: string, lobe: string, mniX: number, mniY: number, mniZ: number, scale = 0.14): AALRegion {
  return { id, name, lobe, hemisphere: 'M', position: [mniX / 50, mniY / 50, mniZ / 50], scale, diseases: LOBE_DISEASE[lobe] };
}

// ── 116 regions ──────────────────────────────────────────────────────────────
// Cortical (90) + Subcortical/Cerebellar (26)
// Positions approximate MNI centroids; scale encodes region volume.

export const AAL_REGIONS: AALRegion[] = [
  // Frontal lobe (24 regions)
  ...lr('Precentral_Gyrus',       'frontal',  37, -15, 54, 0.16),
  ...lr('Frontal_Sup',            'frontal',  23,  49, 32, 0.15),
  ...lr('Frontal_Sup_Orb',        'frontal',  17,  55, -8, 0.12),
  ...lr('Frontal_Mid',            'frontal',  35,  33, 33, 0.14),
  ...lr('Frontal_Mid_Orb',        'frontal',  30,  46,-11, 0.12),
  ...lr('Frontal_Inf_Oper',       'frontal',  54,  12, 16, 0.12),
  ...lr('Frontal_Inf_Tri',        'frontal',  51,  27,  8, 0.12),
  ...lr('Frontal_Inf_Orb',        'frontal',  41,  33,-15, 0.11),
  ...lr('Rolandic_Oper',          'frontal',  54, -10, 14, 0.12),
  ...lr('Supp_Motor_Area',        'frontal',   6,  -4, 63, 0.13),
  ...lr('Frontal_Sup_Med',        'frontal',   8,  57, 28, 0.13),
  ...lr('Frontal_Med_Orb',        'frontal',   6,  52,-10, 0.12),
  // Temporal lobe (16)
  ...lr('Heschl',                 'temporal', 49,  -19, 10, 0.10),
  ...lr('Temporal_Sup',           'temporal', 57,  -22, 6,  0.13),
  ...lr('Temporal_Pole_Sup',      'temporal', 43,   13,-20, 0.11),
  ...lr('Temporal_Mid',           'temporal', 61,  -30,-6,  0.14),
  ...lr('Temporal_Pole_Mid',      'temporal', 42,   11,-34, 0.11),
  ...lr('Temporal_Inf',           'temporal', 54,  -26,-30, 0.13),
  ...lr('ParaHippocampal',        'limbic',   28,  -22,-22, 0.10),
  ...lr('Hippocampus',            'limbic',   29,  -21,-10, 0.10),
  // Parietal lobe (12)
  ...lr('Postcentral',            'parietal', 40,  -27, 50, 0.14),
  ...lr('Parietal_Sup',           'parietal', 25,  -56, 60, 0.13),
  ...lr('Parietal_Inf',           'parietal', 46,  -42, 48, 0.13),
  ...lr('SupraMarginal',          'parietal', 58,  -38, 30, 0.12),
  ...lr('Angular',                'parietal', 47,  -60, 34, 0.12),
  ...lr('Precuneus',              'parietal', 12,  -62, 40, 0.13),
  // Occipital lobe (12)
  ...lr('Calcarine',              'occipital', 11, -73, 10, 0.12),
  ...lr('Cuneus',                 'occipital', 12, -76, 24, 0.11),
  ...lr('Lingual',                'occipital', 16, -69, -4, 0.12),
  ...lr('Occipital_Sup',          'occipital', 22, -77, 24, 0.12),
  ...lr('Occipital_Mid',          'occipital', 33, -81, 13, 0.12),
  ...lr('Occipital_Inf',          'occipital', 36, -79, -7, 0.11),
  // Limbic / cingulate (8)
  ...lr('Cingulum_Ant',           'limbic',    7,  35, 19, 0.11),
  ...lr('Cingulum_Mid',           'limbic',    7, -18, 41, 0.11),
  ...lr('Cingulum_Post',          'limbic',    7, -54, 17, 0.10),
  ...lr('Amygdala',               'limbic',   24,  -5,-20, 0.10),
  // Subcortical (14)
  ...lr('Caudate',                'subcortical', 14,  14,  6, 0.11),
  ...lr('Putamen',                'subcortical', 27,   4,  0, 0.11),
  ...lr('Pallidum',               'subcortical', 20,  -2, -2, 0.09),
  ...lr('Thalamus',               'subcortical', 11, -18,  8, 0.12),
  ...lr('Insula',                 'frontal',     38,  -4, 12, 0.11),
  mid('Olfactory_M',    'Olfactory',   'frontal',   4,  22,-21, 0.09),
  mid('Rectus_M',       'Rectus',      'frontal',   6,  42,-25, 0.09),
  // Cerebellar vermis (10)
  mid('Vermis_1_2',   'Vermis 1-2',   'cerebellum',  0, -58,-28, 0.10),
  mid('Vermis_3',     'Vermis 3',     'cerebellum',  0, -66,-35, 0.10),
  mid('Vermis_4_5',   'Vermis 4-5',   'cerebellum',  0, -70,-40, 0.10),
  mid('Vermis_6',     'Vermis 6',     'cerebellum',  0, -72,-46, 0.10),
  mid('Vermis_7',     'Vermis 7',     'cerebellum',  0, -72,-52, 0.09),
  mid('Vermis_8',     'Vermis 8',     'cerebellum',  0, -70,-58, 0.09),
  mid('Vermis_9',     'Vermis 9',     'cerebellum',  0, -62,-60, 0.09),
  mid('Vermis_10',    'Vermis 10',    'cerebellum',  0, -56,-56, 0.09),
  // Cerebellar hemispheres (18 = 9L + 9R)
  ...lr('Cerebellum_Crus1',  'cerebellum', 37, -66,-34, 0.14),
  ...lr('Cerebellum_Crus2',  'cerebellum', 35, -74,-46, 0.12),
  ...lr('Cerebellum_3',      'cerebellum', 18, -62,-38, 0.11),
  ...lr('Cerebellum_4_5',    'cerebellum', 22, -66,-42, 0.11),
  ...lr('Cerebellum_6',      'cerebellum', 30, -70,-50, 0.11),
  ...lr('Cerebellum_7b',     'cerebellum', 34, -72,-58, 0.10),
  ...lr('Cerebellum_8',      'cerebellum', 30, -68,-62, 0.10),
  ...lr('Cerebellum_9',      'cerebellum', 22, -58,-60, 0.10),
  ...lr('Cerebellum_10',     'cerebellum', 18, -54,-56, 0.09),
].flat();

// Map from feature name to AAL region id (complement to brainAtlas.ts)
export const FEATURE_TO_AAL: Record<string, string> = {
  MMSE:               'Hippocampus_L',
  MemoryComplaints:   'Hippocampus_L',
  delta_hippocampus:  'Hippocampus_L',
  CSF_Abeta42:        'Hippocampus_R',
  CSF_pTau:           'Cingulum_Post_L',
  CSF_tTau:           'Cingulum_Post_R',
  FunctionalAssessment: 'Frontal_Sup_L',
  ADL:                'Frontal_Mid_L',
  BehavioralProblems: 'Frontal_Inf_Oper_L',
  Depression:         'Cingulum_Ant_L',
  PersonalityChanges: 'Frontal_Sup_Med_L',
  UPDRS_motor:        'Putamen_L',
  UPDRS_total:        'Caudate_L',
  tremor_amplitude:   'Putamen_R',
  gait_velocity:      'Cerebellum_4_5_L',
  actigraphy_activity_index: 'Cerebellum_6_L',
  HR_variability:     'Thalamus_L',
  Hypertension:       'Thalamus_R',
  SystolicBP:         'Thalamus_R',
  CardiovascularDisease: 'Vermis_6',
  Diabetes:           'Vermis_7',
  nfl_plasma:         'Frontal_Sup_L',
  APOE4_dosage:       'Temporal_Mid_L',
  polygenetic_risk_score: 'Parietal_Sup_L',
  SpO2_mean:          'Cerebellum_Crus1_L',
  _default:           'Parietal_Sup_L',
};

export function mapFeatureToAAL(featureName: string): string {
  return FEATURE_TO_AAL[featureName] ?? FEATURE_TO_AAL._default;
}
