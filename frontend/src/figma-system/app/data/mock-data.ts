// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
export interface Patient {
  id: string;
  name: string;
  age: number;
  sex: 'M' | 'F';
  mrn: string;
  diagnosis: string;
  deteriorationProb: number;
  riskLevel: 'critical' | 'high' | 'moderate' | 'low';
  lastUpdated: string;
  admissionDate: string;
  ward: string;
  attendingPhysician: string;
}

export interface ForecastPoint {
  time: string;
  predicted: number;
  upper: number;
  lower: number;
  actual?: number;
}

export interface BiomarkerReading {
  time: string;
  heartRate: number;
  spo2: number;
  systolicBP: number;
  diastolicBP: number;
  temperature: number;
  respiratoryRate: number;
}

export interface GenomicRisk {
  gene: string;
  variant: string;
  risk: number;
  confidence: number;
  pathway: string;
}

export interface ConnectomeNode {
  id: string;
  label: string;
  x: number;
  y: number;
  region: string;
  activity: number;
}

export interface ConnectomeEdge {
  source: string;
  target: string;
  weight: number;
}

export interface ClinicalReport {
  id: string;
  generatedAt: string;
  model: string;
  sections: ReportSection[];
}

export interface ReportSection {
  title: string;
  content: string;
  confidence: number;
  evidenceChain: EvidenceItem[];
  claims: Claim[];
}

export interface EvidenceItem {
  source: string;
  type: 'imaging' | 'genomic' | 'lab' | 'wearable' | 'literature';
  description: string;
  timestamp: string;
  confidence: number;
}

export interface Claim {
  text: string;
  confidence: number;
  supporting: number;
  contradicting: number;
}

export interface TimelineEvent {
  id: string;
  timestamp: string;
  modality: 'imaging' | 'genomic' | 'lab' | 'wearable' | 'clinical';
  title: string;
  description: string;
  value?: string;
  flag?: 'normal' | 'abnormal' | 'critical';
}

export const patients: Patient[] = [
  { id: 'P-001', name: 'Nakamura, Kenji', age: 67, sex: 'M', mrn: 'SYN-48291', diagnosis: 'Glioblastoma Multiforme (GBM)', deteriorationProb: 0.89, riskLevel: 'critical', lastUpdated: '2 min ago', admissionDate: '2026-03-15', ward: 'Neuro ICU', attendingPhysician: 'Dr. Vasquez' },
  { id: 'P-002', name: 'Okonkwo, Adaeze', age: 54, sex: 'F', mrn: 'SYN-73104', diagnosis: 'Multiple Sclerosis (RRMS)', deteriorationProb: 0.72, riskLevel: 'high', lastUpdated: '8 min ago', admissionDate: '2026-03-22', ward: 'Neuro 3B', attendingPhysician: 'Dr. Chen' },
  { id: 'P-003', name: 'Petrov, Alexei', age: 71, sex: 'M', mrn: 'SYN-55672', diagnosis: 'Parkinson\'s Disease (Stage 3)', deteriorationProb: 0.61, riskLevel: 'high', lastUpdated: '15 min ago', admissionDate: '2026-02-28', ward: 'Neuro 2A', attendingPhysician: 'Dr. Okafor' },
  { id: 'P-004', name: 'Lindström, Elsa', age: 43, sex: 'F', mrn: 'SYN-91038', diagnosis: 'Epilepsy (TLE)', deteriorationProb: 0.38, riskLevel: 'moderate', lastUpdated: '22 min ago', admissionDate: '2026-04-01', ward: 'EMU', attendingPhysician: 'Dr. Vasquez' },
  { id: 'P-005', name: 'Abadi, Miriam', age: 58, sex: 'F', mrn: 'SYN-24617', diagnosis: 'Alzheimer\'s Disease (Early)', deteriorationProb: 0.29, riskLevel: 'moderate', lastUpdated: '34 min ago', admissionDate: '2026-03-10', ward: 'Memory Clinic', attendingPhysician: 'Dr. Singh' },
  { id: 'P-006', name: 'Torres, Miguel', age: 35, sex: 'M', mrn: 'SYN-68453', diagnosis: 'TBI (Post-Concussive)', deteriorationProb: 0.15, riskLevel: 'low', lastUpdated: '1 hr ago', admissionDate: '2026-04-05', ward: 'Neuro 2A', attendingPhysician: 'Dr. Chen' },
  { id: 'P-007', name: 'Johansson, Erik', age: 62, sex: 'M', mrn: 'SYN-37285', diagnosis: 'ALS (Limb-Onset)', deteriorationProb: 0.82, riskLevel: 'critical', lastUpdated: '5 min ago', admissionDate: '2026-01-20', ward: 'Neuro ICU', attendingPhysician: 'Dr. Okafor' },
  { id: 'P-008', name: 'Gupta, Priya', age: 49, sex: 'F', mrn: 'SYN-50921', diagnosis: 'Myasthenia Gravis', deteriorationProb: 0.11, riskLevel: 'low', lastUpdated: '2 hr ago', admissionDate: '2026-03-28', ward: 'Neuro 3B', attendingPhysician: 'Dr. Singh' },
];

export function generateForecastData(): ForecastPoint[] {
  const data: ForecastPoint[] = [];
  const now = new Date();
  for (let i = -72; i <= 48; i += 4) {
    const t = new Date(now.getTime() + i * 3600000);
    const timeStr = `${String(t.getMonth() + 1).padStart(2,'0')}/${String(t.getDate()).padStart(2,'0')} ${t.getHours().toString().padStart(2, '0')}:${t.getMinutes().toString().padStart(2, '0')}`;
    const base = 0.3 + 0.4 * Math.sin((i + 72) / 30) + (i > 0 ? 0.15 : 0);
    const noise = (Math.random() - 0.5) * 0.08;
    const predicted = Math.max(0, Math.min(1, base + noise));
    const spread = i > 0 ? 0.08 + (i / 48) * 0.12 : 0.05;
    data.push({
      time: timeStr,
      predicted: parseFloat(predicted.toFixed(3)),
      upper: parseFloat(Math.min(1, predicted + spread).toFixed(3)),
      lower: parseFloat(Math.max(0, predicted - spread).toFixed(3)),
      actual: i <= 0 ? parseFloat((predicted + (Math.random() - 0.5) * 0.06).toFixed(3)) : undefined,
    });
  }
  return data;
}

export function generateBiomarkerData(): BiomarkerReading[] {
  const data: BiomarkerReading[] = [];
  for (let i = 0; i < 60; i++) {
    data.push({
      time: `${i}s`,
      heartRate: 72 + Math.sin(i / 5) * 8 + (Math.random() - 0.5) * 4,
      spo2: 96 + Math.sin(i / 8) * 2 + (Math.random() - 0.5) * 1,
      systolicBP: 128 + Math.sin(i / 6) * 10 + (Math.random() - 0.5) * 5,
      diastolicBP: 78 + Math.sin(i / 7) * 6 + (Math.random() - 0.5) * 3,
      temperature: 37.1 + Math.sin(i / 10) * 0.3 + (Math.random() - 0.5) * 0.1,
      respiratoryRate: 16 + Math.sin(i / 4) * 3 + (Math.random() - 0.5) * 2,
    });
  }
  return data;
}

export const genomicRisks: GenomicRisk[] = [
  { gene: 'APOE4', variant: 'rs429358', risk: 0.91, confidence: 0.95, pathway: 'Lipid metabolism' },
  { gene: 'CLU', variant: 'rs11136000', risk: 0.68, confidence: 0.82, pathway: 'Complement cascade' },
  { gene: 'CR1', variant: 'rs6656401', risk: 0.61, confidence: 0.79, pathway: 'Immune regulation' },
  { gene: 'BIN1', variant: 'rs744373', risk: 0.72, confidence: 0.88, pathway: 'Endocytosis' },
  { gene: 'PICALM', variant: 'rs3851179', risk: 0.55, confidence: 0.76, pathway: 'Clathrin assembly' },
  { gene: 'MS4A6A', variant: 'rs610932', risk: 0.48, confidence: 0.71, pathway: 'Membrane transport' },
  { gene: 'ABCA7', variant: 'rs3764650', risk: 0.63, confidence: 0.84, pathway: 'Lipid homeostasis' },
  { gene: 'EPHA1', variant: 'rs11767557', risk: 0.39, confidence: 0.68, pathway: 'Axon guidance' },
  { gene: 'CD33', variant: 'rs3865444', risk: 0.51, confidence: 0.73, pathway: 'Sialic acid binding' },
  { gene: 'TOMM40', variant: 'rs10524523', risk: 0.77, confidence: 0.89, pathway: 'Mitochondrial import' },
  { gene: 'SORL1', variant: 'rs2282649', risk: 0.44, confidence: 0.69, pathway: 'APP trafficking' },
  { gene: 'PTK2B', variant: 'rs28834970', risk: 0.58, confidence: 0.80, pathway: 'Calcium signaling' },
];

export const connectomeNodes: ConnectomeNode[] = [
  { id: 'n1', label: 'Prefrontal Cortex', x: 200, y: 80, region: 'frontal', activity: 0.85 },
  { id: 'n2', label: 'Motor Cortex', x: 280, y: 100, region: 'frontal', activity: 0.62 },
  { id: 'n3', label: 'Somatosensory', x: 340, y: 120, region: 'parietal', activity: 0.45 },
  { id: 'n4', label: 'Visual Cortex', x: 380, y: 220, region: 'occipital', activity: 0.71 },
  { id: 'n5', label: 'Temporal Lobe', x: 140, y: 200, region: 'temporal', activity: 0.93 },
  { id: 'n6', label: 'Hippocampus', x: 260, y: 200, region: 'limbic', activity: 0.88 },
  { id: 'n7', label: 'Amygdala', x: 220, y: 240, region: 'limbic', activity: 0.76 },
  { id: 'n8', label: 'Thalamus', x: 280, y: 170, region: 'subcortical', activity: 0.58 },
  { id: 'n9', label: 'Cerebellum', x: 360, y: 280, region: 'cerebellum', activity: 0.34 },
  { id: 'n10', label: 'Basal Ganglia', x: 240, y: 160, region: 'subcortical', activity: 0.67 },
  { id: 'n11', label: 'Insula', x: 180, y: 160, region: 'limbic', activity: 0.52 },
  { id: 'n12', label: 'Cingulate', x: 240, y: 120, region: 'limbic', activity: 0.79 },
];

export const connectomeEdges: ConnectomeEdge[] = [
  { source: 'n1', target: 'n2', weight: 0.8 },
  { source: 'n1', target: 'n12', weight: 0.9 },
  { source: 'n1', target: 'n10', weight: 0.6 },
  { source: 'n2', target: 'n3', weight: 0.75 },
  { source: 'n2', target: 'n8', weight: 0.5 },
  { source: 'n3', target: 'n4', weight: 0.4 },
  { source: 'n5', target: 'n6', weight: 0.85 },
  { source: 'n5', target: 'n7', weight: 0.7 },
  { source: 'n6', target: 'n7', weight: 0.92 },
  { source: 'n6', target: 'n8', weight: 0.65 },
  { source: 'n6', target: 'n12', weight: 0.78 },
  { source: 'n7', target: 'n11', weight: 0.6 },
  { source: 'n8', target: 'n10', weight: 0.55 },
  { source: 'n8', target: 'n9', weight: 0.3 },
  { source: 'n10', target: 'n12', weight: 0.72 },
  { source: 'n11', target: 'n12', weight: 0.48 },
  { source: 'n1', target: 'n5', weight: 0.55 },
  { source: 'n4', target: 'n8', weight: 0.42 },
];

export const clinicalReport: ClinicalReport = {
  id: 'RPT-2026-04-08-001',
  generatedAt: '2026-04-08T14:32:00Z',
  model: 'NeuroSynth-LLM v3.2.1',
  sections: [
    {
      title: 'Clinical Assessment',
      content: 'Patient Nakamura presents with progressive cognitive decline consistent with glioblastoma-related neurocognitive disorder. fMRI analysis reveals increased BOLD signal asymmetry in the left temporal-parietal junction (TPJ), correlating with worsening language processing deficits observed over the past 14 days. Wearable data indicates a 23% increase in nocturnal heart rate variability (HRV) instability, a known precursor to autonomic dysregulation in GBM patients.',
      confidence: 0.87,
      evidenceChain: [
        { source: 'fMRI Scan #47', type: 'imaging', description: 'BOLD asymmetry index increased from 0.12 to 0.31 in left TPJ', timestamp: '2026-04-06', confidence: 0.91 },
        { source: 'Wearable HRV Stream', type: 'wearable', description: 'Nocturnal RMSSD decreased 23% over 7-day window', timestamp: '2026-04-07', confidence: 0.84 },
        { source: 'Neuropsych Battery', type: 'lab', description: 'Boston Naming Test score declined from 42 to 31', timestamp: '2026-04-05', confidence: 0.93 },
      ],
      claims: [
        { text: 'TPJ BOLD asymmetry correlates with language processing deficits', confidence: 0.87, supporting: 12, contradicting: 2 },
        { text: 'HRV instability is a precursor to autonomic dysregulation', confidence: 0.79, supporting: 8, contradicting: 3 },
        { text: 'Cognitive decline rate exceeds expected progression for tumor grade', confidence: 0.72, supporting: 5, contradicting: 4 },
      ],
    },
    {
      title: 'Causal Pathways',
      content: 'Multi-modal causal analysis identifies three primary deterioration pathways: (1) Tumor-induced white matter disruption → corpus callosum degradation → bilateral motor coordination loss (probability: 0.78). (2) APOE ε4 homozygosity → impaired amyloid-β clearance → accelerated peritumoral edema (probability: 0.65). (3) Sleep fragmentation (wearable-detected) → cortisol dysregulation → blood-brain barrier permeability increase (probability: 0.58).',
      confidence: 0.74,
      evidenceChain: [
        { source: 'DTI Tractography', type: 'imaging', description: 'Fractional anisotropy in corpus callosum decreased 18%', timestamp: '2026-04-04', confidence: 0.88 },
        { source: 'Genomic Panel', type: 'genomic', description: 'APOE ε4/ε4 genotype confirmed with high penetrance', timestamp: '2026-03-15', confidence: 0.96 },
        { source: 'Actigraphy Data', type: 'wearable', description: 'Sleep efficiency dropped from 82% to 61%', timestamp: '2026-04-07', confidence: 0.81 },
        { source: 'Nature Neuroscience 2025', type: 'literature', description: 'Meta-analysis confirms APOE-edema pathway in GBM (n=2,847)', timestamp: '2025-11-14', confidence: 0.85 },
      ],
      claims: [
        { text: 'White matter disruption is the primary driver of motor decline', confidence: 0.78, supporting: 14, contradicting: 3 },
        { text: 'APOE ε4 accelerates peritumoral edema independently of tumor progression', confidence: 0.65, supporting: 7, contradicting: 5 },
        { text: 'Sleep fragmentation causally contributes to BBB permeability via cortisol', confidence: 0.58, supporting: 6, contradicting: 6 },
      ],
    },
    {
      title: 'Intervention Recommendations',
      content: 'Based on the integrated multi-modal analysis, the following interventions are recommended in order of clinical priority: (1) Escalate dexamethasone to 8mg BID to address peritumoral edema (evidence strength: strong). (2) Initiate melatonin 3mg QHS + sleep hygiene protocol to address sleep fragmentation pathway (evidence strength: moderate). (3) Consider bevacizumab addition given BBB permeability indicators and VEGF overexpression pattern (evidence strength: moderate, requires multidisciplinary review). (4) Schedule repeat fMRI in 72 hours to monitor TPJ BOLD trajectory (evidence strength: strong).',
      confidence: 0.81,
      evidenceChain: [
        { source: 'Clinical Guidelines DB', type: 'literature', description: 'NCCN 2026 GBM Guidelines: Dexamethasone escalation criteria met', timestamp: '2026-01-15', confidence: 0.94 },
        { source: 'Cochrane Review', type: 'literature', description: 'Melatonin for sleep in neuro-oncology (moderate evidence)', timestamp: '2025-08-22', confidence: 0.72 },
        { source: 'VEGF Assay', type: 'lab', description: 'Serum VEGF elevated at 892 pg/mL (ref: <500)', timestamp: '2026-04-06', confidence: 0.89 },
      ],
      claims: [
        { text: 'Dexamethasone escalation will reduce edema within 48-72 hours', confidence: 0.88, supporting: 18, contradicting: 1 },
        { text: 'Melatonin supplementation will improve sleep efficiency by >15%', confidence: 0.62, supporting: 5, contradicting: 4 },
        { text: 'Bevacizumab may slow cognitive decline through VEGF pathway inhibition', confidence: 0.55, supporting: 9, contradicting: 7 },
      ],
    },
  ],
};

export const timelineEvents: TimelineEvent[] = [
  { id: 'e1', timestamp: '2026-04-08 14:00', modality: 'wearable', title: 'HR Spike Detected', description: 'Heart rate exceeded 110 bpm for >5 min', value: '118 bpm', flag: 'abnormal' },
  { id: 'e2', timestamp: '2026-04-08 10:30', modality: 'lab', title: 'CRP Result', description: 'C-reactive protein elevated', value: '24.8 mg/L', flag: 'critical' },
  { id: 'e3', timestamp: '2026-04-07 16:00', modality: 'imaging', title: 'fMRI Completed', description: 'Resting-state fMRI, 320 volumes acquired', value: 'See connectome', flag: 'normal' },
  { id: 'e4', timestamp: '2026-04-07 09:00', modality: 'clinical', title: 'Neuropsych Battery', description: 'Boston Naming Test, Trail Making A/B', value: 'BNT: 31/60', flag: 'abnormal' },
  { id: 'e5', timestamp: '2026-04-06 14:00', modality: 'imaging', title: 'DTI Tractography', description: 'Diffusion tensor imaging, corpus callosum focus', value: 'FA: 0.38', flag: 'abnormal' },
  { id: 'e6', timestamp: '2026-04-06 08:00', modality: 'lab', title: 'VEGF Assay', description: 'Serum VEGF quantification', value: '892 pg/mL', flag: 'critical' },
  { id: 'e7', timestamp: '2026-04-05 12:00', modality: 'wearable', title: 'Sleep Report', description: 'Sleep efficiency analysis (7-day)', value: '61%', flag: 'abnormal' },
  { id: 'e8', timestamp: '2026-04-04 10:00', modality: 'genomic', title: 'Panel Update', description: 'Extended neurodegeneration panel results', value: '12 variants', flag: 'normal' },
  { id: 'e9', timestamp: '2026-04-03 14:00', modality: 'clinical', title: 'Medication Adjustment', description: 'Dexamethasone increased to 4mg BID', flag: 'normal' },
  { id: 'e10', timestamp: '2026-04-02 09:00', modality: 'lab', title: 'CBC + Metabolic Panel', description: 'Routine bloodwork, all within normal limits', value: 'WNL', flag: 'normal' },
  { id: 'e11', timestamp: '2026-03-30 16:00', modality: 'imaging', title: 'MRI Brain w/ Contrast', description: 'Surveillance MRI, slight progression noted', value: '+2mm', flag: 'abnormal' },
  { id: 'e12', timestamp: '2026-03-28 10:00', modality: 'wearable', title: 'Gait Analysis', description: 'IMU-based gait assessment', value: 'Stride var: +18%', flag: 'abnormal' },
];

export const forecastData: ForecastPoint[] = [
  { time: 'Apr', predicted: 0.41, upper: 0.52, lower: 0.30, actual: 0.39 },
  { time: 'May', predicted: 0.44, upper: 0.56, lower: 0.32, actual: 0.46 },
  { time: 'Jun', predicted: 0.48, upper: 0.61, lower: 0.35, actual: 0.49 },
  { time: 'Jul', predicted: 0.53, upper: 0.67, lower: 0.39, actual: 0.55 },
  { time: 'Aug', predicted: 0.58, upper: 0.73, lower: 0.43 },
  { time: 'Sep', predicted: 0.63, upper: 0.79, lower: 0.47 },
  { time: 'Oct', predicted: 0.69, upper: 0.85, lower: 0.53 },
  { time: 'Nov', predicted: 0.74, upper: 0.90, lower: 0.58 },
  { time: 'Dec', predicted: 0.78, upper: 0.93, lower: 0.63 },
  { time: 'Jan', predicted: 0.81, upper: 0.95, lower: 0.67 },
  { time: 'Feb', predicted: 0.83, upper: 0.96, lower: 0.70 },
  { time: 'Mar', predicted: 0.86, upper: 0.97, lower: 0.75 },
];

export const biomarkerHistory: BiomarkerReading[] = Array.from({ length: 30 }, (_, i) => ({
  time: new Date(Date.now() - (30 - i) * 2000).toLocaleTimeString(),
  heartRate: 72 + Math.sin(i * 0.4) * 8 + Math.random() * 3,
  spo2: 97.5 - Math.abs(Math.sin(i * 0.3)) * 1.5,
  systolicBP: 128 + Math.sin(i * 0.2) * 6,
  diastolicBP: 82 + Math.sin(i * 0.15) * 4,
  temperature: 37.1 + Math.sin(i * 0.1) * 0.2,
  respiratoryRate: 16 + Math.sin(i * 0.35) * 2,
}));

export const connectomeData = {
  nodes: [
    { id: 'PFC', label: 'PFC', region: 'Frontal', activity: 0.82, x: 0, y: 0 },
    { id: 'HC', label: 'Hippocampus', region: 'Temporal', activity: 0.91, x: 0, y: 0 },
    { id: 'AMY', label: 'Amygdala', region: 'Temporal', activity: 0.78, x: 0, y: 0 },
    { id: 'THAL', label: 'Thalamus', region: 'Subcortical', activity: 0.55, x: 0, y: 0 },
    { id: 'CEREB', label: 'Cerebellum', region: 'Posterior', activity: 0.33, x: 0, y: 0 },
    { id: 'BG', label: 'Basal Ganglia', region: 'Subcortical', activity: 0.62, x: 0, y: 0 },
    { id: 'ACC', label: 'ACC', region: 'Cingulate', activity: 0.74, x: 0, y: 0 },
    { id: 'INS', label: 'Insula', region: 'Lateral', activity: 0.69, x: 0, y: 0 },
  ] as ConnectomeNode[],
  edges: [
    { source: 'PFC', target: 'HC', weight: 0.85 },
    { source: 'HC', target: 'AMY', weight: 0.72 },
    { source: 'PFC', target: 'ACC', weight: 0.91 },
    { source: 'THAL', target: 'PFC', weight: 0.67 },
    { source: 'THAL', target: 'BG', weight: 0.58 },
    { source: 'AMY', target: 'INS', weight: 0.63 },
    { source: 'ACC', target: 'INS', weight: 0.77 },
    { source: 'BG', target: 'CEREB', weight: 0.44 },
    { source: 'HC', target: 'THAL', weight: 0.69 },
    { source: 'PFC', target: 'BG', weight: 0.55 },
  ] as ConnectomeEdge[],
};