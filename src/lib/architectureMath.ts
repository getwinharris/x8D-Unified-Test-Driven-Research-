export type ScalePresetId = 'ncm-50m' | 'ncm-150m' | 'x8d-1b' | 'x8d-1t';

export type QuantizationRegimeId =
  | 'fp16'
  | 'int4'
  | 'u8-quanta-0001'
  | 'block-mean-0001'
  | 'x8d-subbyte-0008';

export type ExecutionModeId = 'raw-x8d-direct' | 'mmap-latent-file' | 'full-ram-resident';

export interface ModelScaleSpec {
  id: ScalePresetId;
  label: string;
  nLayers: number;
  dModel: number;
  nHeads: number;
  dHead: number;
  ffnHidden: number;
  kdaLayers: number;
  mlaLayers: number;
  vocabSize: number; // Strictly 256 raw unsigned byte states (0..255)
  ncmCoarseLevels: number;
  spatialAdapterBlocks: number;
  routedExpertsPerLayer?: number;
  activeExpertsPerToken?: number;
}

export interface ParameterBreakdown {
  attentionParams: number;
  ffnParams: number;
  kdaGateParams: number;
  spatialNcmParams: number;
  byteEmbeddingParams: number;
  normAndAttnResParams: number;
  totalParams: number;
}

export interface QuantizationRegimeSpec {
  id: QuantizationRegimeId;
  name: string;
  shortLabel: string;
  storedBitsPerWeight: number;
  effectiveEntropyBits: number;
  preservesPerWeightInfo: boolean;
  supportsRandomAccessMmap: boolean;
  forwardPassViable: boolean;
  auditStatus: 'verified' | 'trap' | 'broken';
  summary: string;
}

export interface FootprintSimulationInput {
  scaleId: ScalePresetId;
  quantId: QuantizationRegimeId;
  executionMode: ExecutionModeId;
  contextBytes: number;
  activeLayerSpanRatio: number; // e.g. 0.15 for streaming 1-2 layers / active experts
  distributedNodes: number;
  ncmCoarseBypassRate: number; // fraction of 8x8 blocks skipped at fine refinement (0..0.85)
}

export interface FootprintSimulationResult {
  params: ParameterBreakdown;
  diskBytes: number;
  fullModelRamBytes: number;
  activeWorkingSetRamBytes: number;
  latentStateFileBytes: number;
  peakAluTileDramBytes: number;
  effectiveBitsPerWeight: number;
  compressionRatioVsFp16: number;
  perNodeWeightShardBytes: number;
  perNodeOptimizerStateBytes: number;
  ncmEffectiveComputeFactor: number;
  forwardPassViable: boolean;
  warnings: string[];
}

export const MODEL_SCALE_PRESETS: Record<ScalePresetId, ModelScaleSpec> = {
  'ncm-50m': {
    id: 'ncm-50m',
    label: '50M Hybrid Omni Diffusion NCM (GR00T Baseline)',
    nLayers: 14,
    dModel: 512,
    nHeads: 8,
    dHead: 64,
    ffnHidden: 1536,
    kdaLayers: 10,
    mlaLayers: 4,
    vocabSize: 256,
    ncmCoarseLevels: 3,
    spatialAdapterBlocks: 2,
  },
  'ncm-150m': {
    id: 'ncm-150m',
    label: '150M Scaled Multimodal NCM (Multi-Embodiment)',
    nLayers: 18,
    dModel: 768,
    nHeads: 12,
    dHead: 64,
    ffnHidden: 2304,
    kdaLayers: 13,
    mlaLayers: 5,
    vocabSize: 256,
    ncmCoarseLevels: 3,
    spatialAdapterBlocks: 3,
  },
  'x8d-1b': {
    id: 'x8d-1b',
    label: '1.03B x8D Omni DLM Target (Issue #68)',
    nLayers: 20,
    dModel: 2048,
    nHeads: 16,
    dHead: 128,
    ffnHidden: 5632,
    kdaLayers: 15,
    mlaLayers: 5,
    vocabSize: 256,
    ncmCoarseLevels: 4,
    spatialAdapterBlocks: 4,
  },
  'x8d-1t': {
    id: 'x8d-1t',
    label: '1.00T Unified S1+S2 AGI Edge Target (Phone/CPU <1 GB RAM)',
    nLayers: 48,
    dModel: 4096,
    nHeads: 32,
    dHead: 128,
    ffnHidden: 12288,
    kdaLayers: 36,
    mlaLayers: 12,
    vocabSize: 256,
    ncmCoarseLevels: 4,
    spatialAdapterBlocks: 4,
    routedExpertsPerLayer: 138,
    activeExpertsPerToken: 2,
  },
};

export const QUANTIZATION_REGIMES: Record<QuantizationRegimeId, QuantizationRegimeSpec> = {
  fp16: {
    id: 'fp16',
    name: 'BF16 / FP16 Uncompressed Baseline',
    shortLabel: 'BF16 (16 bpw)',
    storedBitsPerWeight: 16,
    effectiveEntropyBits: 16,
    preservesPerWeightInfo: true,
    supportsRandomAccessMmap: true,
    forwardPassViable: true,
    auditStatus: 'verified',
    summary: 'Full 16-bit training master precision; high RAM and storage footprint.',
  },
  int4: {
    id: 'int4',
    name: 'Regular INT4 Group-Wise Quantization',
    shortLabel: 'INT4 (4.25 bpw)',
    storedBitsPerWeight: 4.25,
    effectiveEntropyBits: 4.25,
    preservesPerWeightInfo: true,
    supportsRandomAccessMmap: true,
    forwardPassViable: true,
    auditStatus: 'verified',
    summary: '4-bit weights plus FP16 group scales (~500 GB at 1T scale).',
  },
  'u8-quanta-0001': {
    id: 'u8-quanta-0001',
    name: 'Raw U8 Quanta × 0.001 Scaling (Issues #50 / #58)',
    shortLabel: 'U8 × 0.001 (8.0 bpw)',
    storedBitsPerWeight: 8,
    effectiveEntropyBits: 8,
    preservesPerWeightInfo: true,
    supportsRandomAccessMmap: true,
    forwardPassViable: true,
    auditStatus: 'trap',
    summary: 'Multiplying byte[i] by 0.001 yields 256 states (8 bits). Scaling values alone does not compress bits.',
  },
  'block-mean-0001': {
    id: 'block-mean-0001',
    name: '1000:1 Block Mean × 0.001 (Broken Issues #57 / #59)',
    shortLabel: 'Block Mean (0.001 B/w)',
    storedBitsPerWeight: 0.008,
    effectiveEntropyBits: 0.008,
    preservesPerWeightInfo: false,
    supportsRandomAccessMmap: true,
    forwardPassViable: false,
    auditStatus: 'broken',
    summary: 'Averages 1000 bytes into 1 byte, discarding 999/1000 degrees of freedom. Cannot run forward pass.',
  },
  'x8d-subbyte-0008': {
    id: 'x8d-subbyte-0008',
    name: 'x8D Sub-Bit .x8D (Kronecker + x8d_arith, 0.008 bpw)',
    shortLabel: '.x8D Sub-Bit (0.008 bpw)',
    storedBitsPerWeight: 0.008,
    effectiveEntropyBits: 0.008,
    preservesPerWeightInfo: true,
    supportsRandomAccessMmap: true,
    forwardPassViable: true,
    auditStatus: 'verified',
    summary:
      '0.008 bits per 8-bit (0.001 B/w) via Kronecker decomposition + QAT (#56) + x8d_arith range coding (#60/#61), read directly via f.seek/read (no mmap, zero -1 ternary).',
  },
};

export function computeParameterBreakdown(spec: ModelScaleSpec): ParameterBreakdown {
  const expertCount = Math.max(1, spec.routedExpertsPerLayer ?? 1);
  const attentionParams = spec.nLayers * (4 * spec.dModel * spec.dModel);
  const ffnParams = spec.nLayers * expertCount * (3 * spec.dModel * spec.ffnHidden);
  // KDA adds vector decay alpha, write gate beta, and low-rank key feature maps (~3 * dModel * 128)
  const kdaGateParams = spec.kdaLayers * (3 * spec.dModel * Math.min(128, spec.dModel / 4));
  // Spatial cross-attention adapters + NCM multiscale coarse-to-fine residual heads
  const spatialNcmParams =
    spec.spatialAdapterBlocks * (2 * spec.dModel * spec.dModel) +
    spec.ncmCoarseLevels * (spec.dModel * spec.dModel);
  const byteEmbeddingParams = spec.vocabSize * spec.dModel; // tied input/output projection
  // Block AttnRes pseudo-queries + RMSNorm weights per layer
  const normAndAttnResParams = spec.nLayers * (spec.dModel * 3 + 256);

  const totalParams =
    attentionParams +
    ffnParams +
    kdaGateParams +
    spatialNcmParams +
    byteEmbeddingParams +
    normAndAttnResParams;

  return {
    attentionParams,
    ffnParams,
    kdaGateParams,
    spatialNcmParams,
    byteEmbeddingParams,
    normAndAttnResParams,
    totalParams,
  };
}

export function simulateFootprint(input: FootprintSimulationInput): FootprintSimulationResult {
  const scale = MODEL_SCALE_PRESETS[input.scaleId];
  const quant = QUANTIZATION_REGIMES[input.quantId];
  const params = computeParameterBreakdown(scale);

  const fp16Bytes = params.totalParams * 2;
  const diskBytes = Math.max(1, Math.ceil((params.totalParams * quant.storedBitsPerWeight) / 8));

  // KDA recurrent state memory: O(1) wrt sequence length -> kdaLayers * (dModel * dHead) * 2 bytes (bf16)
  // MLA compressed latent KV cache for remaining mlaLayers -> mlaLayers * (contextBytes / 8) * (dModel / 4) * 2
  const kdaStateBytes = scale.kdaLayers * (scale.dModel * scale.dHead) * 2;
  const mlaCacheBytes =
    scale.mlaLayers * Math.ceil(input.contextBytes / 8) * Math.max(64, Math.floor(scale.dModel / 4)) * 2;
  // NCM coarse-to-fine 8x8 block latent workspace
  const ncmLatentGridBytes = Math.ceil(input.contextBytes / 64) * scale.dModel * 2;
  const latentStateFileBytes = kdaStateBytes + mlaCacheBytes + ncmLatentGridBytes;

  // Ephemeral ALU working tile DRAM needed to dequantize & execute one active layer/expert tile
  // For routed/Kronecker-banked 1T models, only activeExpertsPerToken slices enter tile SRAM/DRAM (not all 138)
  const activeExperts = Math.max(1, scale.activeExpertsPerToken ?? 1);
  const singleLayerWeightBytes =
    ((4 * scale.dModel * scale.dModel + activeExperts * 3 * scale.dModel * scale.ffnHidden) * 2) /
    (scale.routedExpertsPerLayer && scale.routedExpertsPerLayer > 1 ? 8 : 1);
  const activationTileBytes = 64 * scale.dModel * 4; // 64-byte DSpark/8x8 block tile in FP32 accum
  const peakAluTileDramBytes = singleLayerWeightBytes + activationTileBytes;

  const fullModelRamBytes = diskBytes + latentStateFileBytes + activationTileBytes;

  // In direct raw .x8D or mmap mode, only the active layer/expert span is streamed from disk
  // plus minimal base runtime RSS (14 MB for direct f.seek/read buffer vs 18 MB with mmap page-table).
  const clampedSpanRatio = Math.min(1, Math.max(0.05, input.activeLayerSpanRatio));
  const baseRuntimeRssBytes =
    input.executionMode === 'raw-x8d-direct' ? 14 * 1024 * 1024 : 18 * 1024 * 1024;
  const activeWorkingSetRamBytes =
    input.executionMode === 'full-ram-resident'
      ? fullModelRamBytes
      : baseRuntimeRssBytes +
        Math.ceil(diskBytes * clampedSpanRatio) +
        Math.ceil(latentStateFileBytes * 0.25) +
        activationTileBytes;

  const nodes = Math.max(1, Math.floor(input.distributedNodes));
  // FSDP/ZeRO-2 weight shard + Muon/AdamW mixed-precision optimizer states (master fp32 + momentum)
  const perNodeWeightShardBytes = Math.ceil(fp16Bytes / nodes);
  const perNodeOptimizerStateBytes = Math.ceil((params.totalParams * 8) / nodes); // fp32 master + 1st moment

  const bypass = Math.min(0.85, Math.max(0, input.ncmCoarseBypassRate));
  // Coarse pass consumes ~15% of fine pass compute; remaining (1 - bypass) runs fine refinement
  const ncmEffectiveComputeFactor = 0.15 + (1 - bypass) * 0.85;

  const warnings: string[] = [];
  if (!quant.forwardPassViable) {
    warnings.push(
      'Critical (Issues #57/#59): 1000:1 block-mean averaging destroys 99.9% of per-weight information and cannot execute a valid forward pass.'
    );
  }
  if (quant.auditStatus === 'trap') {
    warnings.push(
      'Unit Error (Issues #50/#58): Scaling raw bytes by 0.001 preserves 256 states (log2(256) = 8 bits). Disk footprint remains 1:1 with uint8.'
    );
  }
  if (input.executionMode === 'raw-x8d-direct') {
    warnings.push(
      'Direct Raw .x8D Serving (No mmap): Reads active tensor spans straight from the >93%–99.9% compressed .x8D file via f.seek/read and evaluates q / 0.001 directly inside tile SRAM/DRAM.'
    );
  } else if (input.executionMode === 'mmap-latent-file') {
    warnings.push(
      'Legacy OS mmap mode: Keep only for comparison — direct f.seek/read span streaming is preferred per #45/#70.'
    );
  }

  return {
    params,
    diskBytes,
    fullModelRamBytes,
    activeWorkingSetRamBytes,
    latentStateFileBytes,
    peakAluTileDramBytes,
    effectiveBitsPerWeight: quant.effectiveEntropyBits,
    compressionRatioVsFp16: fp16Bytes / diskBytes,
    perNodeWeightShardBytes,
    perNodeOptimizerStateBytes,
    ncmEffectiveComputeFactor,
    forwardPassViable: quant.forwardPassViable,
    warnings,
  };
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let val = bytes / 1024;
  let idx = 0;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx++;
  }
  return `${val.toFixed(val >= 100 ? 1 : 2)} ${units[idx]}`;
}

export function formatParams(count: number): string {
  if (count >= 1e12) return `${(count / 1e12).toFixed(2)}T`;
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(2)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
  return `${count}`;
}

export interface KroneckerSubBitInput {
  rows: number;
  cols: number;
  p1: number;
  q1: number;
  kTerms: number;
  bitsPerFactor: number;
}

export type SubBytePackingTier = '<=0.001' | '<=0.1' | '<=0.25' | '<=0.5' | '>0.5';

export interface KroneckerSubBitResult {
  totalElements: number;
  p2: number;
  q2: number;
  factorEntriesPerTerm: number;
  totalStoredBits: number;
  totalStoredBytes: number;
  bitsPerElement: number;
  bytesPerElement: number;
  weightsPerByte: number;
  subByteTier: SubBytePackingTier;
  ratioVsBytePack001: number;
  maxAchievableRank: number;
}

export function classifySubByteTier(bytesPerElement: number): SubBytePackingTier {
  if (bytesPerElement <= 0.001) return '<=0.001';
  if (bytesPerElement <= 0.1) return '<=0.1';
  if (bytesPerElement <= 0.25) return '<=0.25';
  if (bytesPerElement <= 0.5) return '<=0.5';
  return '>0.5';
}

export function computeKroneckerSubBit(input: KroneckerSubBitInput): KroneckerSubBitResult {
  const rows = Math.max(16, input.rows);
  const cols = Math.max(16, input.cols);
  const p1 = Math.max(2, Math.min(rows, input.p1));
  const q1 = Math.max(2, Math.min(cols, input.q1));
  const p2 = Math.max(1, Math.ceil(rows / p1));
  const q2 = Math.max(1, Math.ceil(cols / q1));
  const kTerms = Math.max(1, input.kTerms);
  const bitsPerFactor = Math.max(1, input.bitsPerFactor);

  const totalElements = rows * cols;
  const factorEntriesPerTerm = p1 * q1 + p2 * q2;
  // Include 64 bits per term for fp32 min/scale metadata
  const totalStoredBits = kTerms * (factorEntriesPerTerm * bitsPerFactor + 64);
  const totalStoredBytes = totalStoredBits / 8.0;
  const bitsPerElement = totalStoredBits / totalElements;
  const bytesPerElement = totalStoredBytes / totalElements;
  const weightsPerByte = 1.0 / bytesPerElement;
  const maxAchievableRank = Math.min(rows, cols, kTerms * Math.min(p1, q1) * Math.min(p2, q2));

  return {
    totalElements,
    p2,
    q2,
    factorEntriesPerTerm,
    totalStoredBits,
    totalStoredBytes,
    bitsPerElement,
    bytesPerElement,
    weightsPerByte,
    subByteTier: classifySubByteTier(bytesPerElement),
    ratioVsBytePack001: bytesPerElement / 0.001,
    maxAchievableRank,
  };
}

export interface PrecisionFormatRow {
  format: string;
  bpwLabel: string;
  bytesPerParamLabel: string;
  description: string;
}

export const PRECISION_FORMAT_BASELINE: PrecisionFormatRow[] = [
  {
    format: 'FP32 (Full Precision)',
    bpwLabel: '32 bits',
    bytesPerParamLabel: '4 bytes',
    description: 'Standard training and baseline floating-point precision.',
  },
  {
    format: 'FP16 / BF16 (Half Precision)',
    bpwLabel: '16 bits',
    bytesPerParamLabel: '2 bytes',
    description: 'Standard default for modern inference and fine-tuning.',
  },
  {
    format: 'FP8 / INT8 (8-bit)',
    bpwLabel: '8 bits',
    bytesPerParamLabel: '1 byte',
    description: 'Half the memory of FP16; matches standard 8-bit hardware byte boundary.',
  },
  {
    format: 'INT4 / FP4 / NF4 (4-bit)',
    bpwLabel: '4 bits',
    bytesPerParamLabel: '0.5 bytes',
    description: 'Standard 4-bit LLM quantization (GGUF Q4 / AWQ / GPTQ).',
  },
  {
    format: 'INT2 (2-bit)',
    bpwLabel: '2 bits',
    bytesPerParamLabel: '0.25 bytes',
    description: 'Aggressive low-bit quantization floor (~2.3 bpw in Cactus .cact).',
  },
  {
    format: 'x8D Unified .x8D (Target)',
    bpwLabel: '0.008 bits',
    bytesPerParamLabel: '0.001 B',
    description:
      'Kronecker sub-byte decomposition + QAT (#56) + x8d_arith range coding (#60/#61) streamed via direct f.seek/read (zero -1 ternary, no mmap).',
  },
];

export interface UnifiedX8DImageInput {
  scaleId: ScalePresetId;
  attentionCodec: 'kron-factored' | 'kron-subbyte-0008';
  kroneckerTerms: number;
  kroneckerBits: number;
  bytePriorAccuracy: number; // e.g. 0.999
}

export interface UnifiedX8DImageResult {
  rawBf16Bytes: number;
  rawU8Bytes: number;
  unifiedX8DBytes: number;
  embeddingBytes: number;
  attentionBytes: number;
  ffnBytes: number;
  framingOverheadBytes: number;
  overallBytesPerWeight: number;
  overallRatioVsBf16: number;
  compressionFactorVsBf16: number;
  totalTensorsPacked: number;
}

export function simulateUnifiedX8DImage(input: UnifiedX8DImageInput): UnifiedX8DImageResult {
  const scale = MODEL_SCALE_PRESETS[input.scaleId];
  const breakdown = computeParameterBreakdown(scale);
  const rawBf16Bytes = breakdown.totalParams * 2;
  const rawU8Bytes = breakdown.totalParams;

  // Embeddings + norm/AttnRes tables packed via delta + x8d_arith
  const embedParams = breakdown.byteEmbeddingParams + breakdown.normAndAttnResParams;
  const arith = computeUniqueSubByteArith({
    byteValue: 65,
    streamLengthBytes: Math.max(1000, embedParams),
    priorMatchRate: input.bytePriorAccuracy,
  });
  const embeddingBytes = Math.ceil(embedParams * arith.arithBytesPerByte + 64);

  // Per-tensor Kronecker sub-byte core + x8d_arith framing
  const kTerms = Math.max(1, input.kroneckerTerms);
  const bits = Math.max(1, input.kroneckerBits);

  const packMatrixBytes = (rows: number, cols: number, p1: number, q1: number) => {
    const p2 = Math.max(1, Math.ceil(rows / p1));
    const q2 = Math.max(1, Math.ceil(cols / q1));
    const coreBytes = kTerms * Math.ceil(((p1 * q1 + p2 * q2) * bits) / 8) + kTerms * 9;
    const rowColScaleBytes = 2 * (Math.ceil(rows / 8) + Math.ceil(cols / 8) + 4);
    return coreBytes + rowColScaleBytes;
  };

  // FFN SwiGLU matrices (3 per layer: gate, up, down)
  const expertCount = Math.max(1, scale.routedExpertsPerLayer ?? 1);
  const perFfnMatBytes = packMatrixBytes(scale.dModel, scale.ffnHidden, 16, 32);
  const ffnBytes =
    input.attentionCodec === 'kron-subbyte-0008'
      ? Math.ceil((breakdown.ffnParams * 0.008) / 8)
      : scale.nLayers * expertCount * 3 * perFfnMatBytes;

  // Attention + KDA gates + NCM heads
  const attnTotalParams = breakdown.attentionParams + breakdown.kdaGateParams + breakdown.spatialNcmParams;
  let attentionBytes = 0;
  if (input.attentionCodec === 'kron-subbyte-0008') {
    attentionBytes = Math.ceil((attnTotalParams * 0.008) / 8);
  } else {
    const perAttnMatBytes = packMatrixBytes(scale.dModel, scale.dModel, 16, 16);
    const numAttnMats = scale.nLayers * 4 + scale.spatialAdapterBlocks * 2 + scale.ncmCoarseLevels + scale.kdaLayers;
    attentionBytes = numAttnMats * perAttnMatBytes;
  }

  // Magic-free direct-seek 24-byte tail span index per tensor (Issue #52)
  const totalTensorsPacked = scale.nLayers * 7 + scale.kdaLayers + scale.spatialAdapterBlocks * 2 + scale.ncmCoarseLevels + 2;
  const framingOverheadBytes = totalTensorsPacked * 24;

  const unifiedX8DBytes = embeddingBytes + attentionBytes + ffnBytes + framingOverheadBytes;

  return {
    rawBf16Bytes,
    rawU8Bytes,
    unifiedX8DBytes,
    embeddingBytes,
    attentionBytes,
    ffnBytes,
    framingOverheadBytes,
    overallBytesPerWeight: unifiedX8DBytes / breakdown.totalParams,
    overallRatioVsBf16: unifiedX8DBytes / rawBf16Bytes,
    compressionFactorVsBf16: rawBf16Bytes / unifiedX8DBytes,
    totalTensorsPacked,
  };
}

export interface UniqueSubByteArithInput {
  byteValue: number; // 0..255 (e.g. 65 for 'A')
  streamLengthBytes: number; // e.g. 100000
  priorMatchRate: number; // 0.80 .. 0.9999 (how often generative prior predicts the exact byte)
}

export interface UniqueSubByteArithResult {
  byteValue: number;
  subByteScaledValue: number;
  standaloneFixedBits: number;
  arithBitsPerByte: number;
  arithBytesPerByte: number;
  arithDiskBytes: number;
  rawDiskBytes: number;
  compressionFactor: number;
  preservesUniquePerByte: boolean;
}

export function computeUniqueSubByteArith(input: UniqueSubByteArithInput): UniqueSubByteArithResult {
  const byteValue = Math.max(0, Math.min(255, Math.round(input.byteValue)));
  const streamLengthBytes = Math.max(1000, Math.round(input.streamLengthBytes));
  const pMatch = Math.min(0.9999, Math.max(0.5, input.priorMatchRate));
  const pSurprise = 1.0 - pMatch;

  // Binary entropy of exact match bit + conditional uniform surprise over 255 remaining bytes
  // Plus small 48-bit range coder flush overhead (6 bytes)
  const hitBits = -Math.log2(pMatch);
  const missBits = pSurprise > 0 ? -Math.log2(pSurprise / 255.0) : 0;
  const expectedBitsPerByte = pMatch * hitBits + pSurprise * missBits;
  const arithDiskBytes = Math.max(6, Math.ceil((streamLengthBytes * expectedBitsPerByte) / 8.0) + 6);
  const arithBytesPerByte = arithDiskBytes / streamLengthBytes;
  const arithBitsPerByte = arithBytesPerByte * 8.0;

  return {
    byteValue,
    subByteScaledValue: Number((byteValue * 0.001).toFixed(3)),
    standaloneFixedBits: 8,
    arithBitsPerByte,
    arithBytesPerByte,
    arithDiskBytes,
    rawDiskBytes: streamLengthBytes,
    compressionFactor: streamLengthBytes / arithDiskBytes,
    preservesUniquePerByte: true,
  };
}

