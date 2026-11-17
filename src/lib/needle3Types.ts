export type Needle3TensorId =
  | 'self_attn_out_proj'
  | 'self_attn_gate_proj'
  | 'engrams_0_key_proj'
  | 'embedding_slice';

export interface Needle3TensorSpec {
  id: Needle3TensorId;
  safetensorsKey: string;
  label: string;
  rows: number;
  cols: number;
  category: 'attention' | 'engram' | 'embedding';
}

export interface Needle3QuantRow {
  methodId: string;
  label: string;
  family: 'baseline' | 'regular' | 'cactus' | 'x8d';
  bpw: number;
  projectedNeedle3Mb: number;
  cosineSimilarity: number;
  sqnrDb: number;
  rmse: number;
  maxAbsErr: number;
  outputActCosine: number;
  logitKlDivergence: number;
  x8dSeekRoundtripVerified: boolean;
  verdictNote: string;
}

export interface Needle3BenchmarkReport {
  modelId: string;
  totalParameters: number;
  safetensorsBytes: number;
  officialCactBytes: number;
  officialCactBpw: number;
  tensor: Needle3TensorSpec;
  tensorStats: {
    min: number;
    max: number;
    rms: number;
    nearZeroPct: number;
  };
  benchmarkDurationMs: number;
  rows: Needle3QuantRow[];
  architectTakeaway: string;
  generatedAtIso: string;
}

export const NEEDLE3_TENSOR_SPECS: Record<Needle3TensorId, Needle3TensorSpec> = {
  self_attn_out_proj: {
    id: 'self_attn_out_proj',
    safetensorsKey: 'stack/layers/block/self_attn/out_proj/kernel',
    label: 'Layer-0 Self-Attn out_proj [768×768]',
    rows: 768,
    cols: 768,
    category: 'attention',
  },
  self_attn_gate_proj: {
    id: 'self_attn_gate_proj',
    safetensorsKey: 'stack/layers/block/self_attn/gate_proj/kernel',
    label: 'Layer-0 Self-Attn gate_proj [768×768]',
    rows: 768,
    cols: 768,
    category: 'attention',
  },
  engrams_0_key_proj: {
    id: 'engrams_0_key_proj',
    safetensorsKey: 'engrams_0/key_proj/kernel',
    label: 'Engram-0 key_proj [768×768]',
    rows: 768,
    cols: 768,
    category: 'engram',
  },
  embedding_slice: {
    id: 'embedding_slice',
    safetensorsKey: 'embedding/embedding',
    label: 'Token Embedding [512×768 slice]',
    rows: 512,
    cols: 768,
    category: 'embedding',
  },
};
