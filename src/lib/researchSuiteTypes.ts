/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared pure-TypeScript interfaces for the x8D Unified Omni-Diffusion
 * Research Brief experiments (browser-safe; zero Node built-in imports).
 */

export interface FixedPointComparisonRow {
  method: string;
  formula: string;
  testedValues: number;
  collisionCount: number;
  maxRoundtripErrorBytes: number;
  exactBijectionPct: number;
  verdict: string;
}

export interface CodecPayloadAuditRow {
  payloadId: string;
  label: string;
  category:
    | 'vocab-256'
    | 'ascii-text'
    | 'tamil-utf8'
    | 'structured-repeat'
    | 'high-entropy-random'
    | 'png-binary'
    | 'wav-binary'
    | 'xat-65536';
  rawBytes: number;
  shannonEntropyBitsPerByte: number;
  physicalX8dBytes: number;
  payloadSpanBytes: number;
  containerOverheadBytes: number;
  codecUsed: 'delta-rle' | 'deflate-frame' | 'raw-verbatim';
  compressionRatio: number;
  storedBitsPerByte: number;
  reverseMatchPct: number;
  sha256Original: string;
  sha256Decoded: string;
  sha256ExactMatch: boolean;
  crc32Verified: boolean;
  encodeMs: number;
  decodeMs: number;
  honestNote: string;
}

export interface FourRegimeRow {
  regimeId: 'lossless-byte-io' | 'lossless-checkpoint' | 'lossy-kronecker' | 'native-compact-ste';
  label: string;
  domain: string;
  measuredRate: string;
  exactLossless: boolean;
  reconstructionMetric: string;
  scopeBoundary: string;
}

export interface ReversibleCodecReport {
  generatedAtIso: string;
  artifactDir: string;
  fixedPointComparison: FixedPointComparisonRow[];
  payloadAudits: CodecPayloadAuditRow[];
  fourRegimes: FourRegimeRow[];
  allPayloadsBitExact: boolean;
  pngValidHeaderAndChunks: boolean;
  wavValidRiffHeader: boolean;
  honestFinding: string;
}

export interface SequenceMethodEvalRow {
  methodId: 'dense-65536' | 'fixed-8x8-legacy' | 'adaptive-entropy-cache' | 'diamond-chain-spine';
  label: string;
  sequenceCapacityBytes: number;
  spansCreated: number;
  activeBytesEvaluated: number;
  activeRatioPct: number;
  indexMetadataBytes: number;
  workingBufferBytes: number;
  latencyMs: number;
  needleRetrievalAccPct: number;
  boundaryAlignmentF1Pct: number;
  verdict: string;
}

export interface XAtSequenceReport {
  generatedAtIso: string;
  sequenceBytes: number;
  embeddedNeedles: number;
  clarificationNote: string;
  rows: SequenceMethodEvalRow[];
  honestHypothesisAudit: string;
}

export interface Mp4ExecutionModeRow {
  modeId:
    | 'dense-fp32-resident'
    | 'compressed-to-full-ram'
    | 'chunk-tile-reconstruct'
    | 'direct-factor-math'
    | 'selective-expert-seek';
  label: string;
  effectiveWeights: number;
  diskFileBytes: number;
  diskReadsCount: number;
  bytesReadFromDisk: number;
  peakWorkingBufferBytes: number;
  materializedDenseMatrix: boolean;
  latencyMs: number;
  maxAbsDiffVsFactorDense: number;
  cosineVsFactorDense: number;
  speedupVsFullMat: number;
}

export interface Mp4AnalogyReport {
  generatedAtIso: string;
  matrixSpec: string;
  numExperts: number;
  activeExpertIndex: number;
  rows: Mp4ExecutionModeRow[];
  exactAlgebraNote: string;
}

export interface NcmStageMetricRow {
  stageId: 'stage1-bytes' | 'stage2-symbolic' | 'stage3-png-wav' | 'stage4-s1-s2' | 'stage5-reload-eval';
  stageLabel: string;
  heldOutBytes: number;
  flatBaselineTop1Pct: number;
  flatBaselinePpl: number;
  ncmUnifiedTop1Pct: number;
  ncmUnifiedPpl: number;
  deltaAccPct: number;
  system1ExitPct: number;
  avgDenoiseSteps: number;
}

export interface NcmModelSummary {
  modelId: 'flat-diffusion-baseline' | 'ncm-rvq-unified';
  label: string;
  effectiveWeights: number;
  trainableDofs: number;
  physicalX8dBytes: number;
  codebookBytes: number;
  measuredBpw: number;
  trainSteps: number;
  trainTimeMs: number;
  reloadParityMaxDiff: number;
  overallHeldOutPpl: number;
  overallTop1AccPct: number;
  coarseLevelOnlyAccPct: number;
  coarsePlusFineAccPct: number;
  avgStepsPerSpan: number;
  evalLatencyMs: number;
  evalBufferBytes: number;
  sampleOutputs: Array<{ modality: string; prompt: string; generatedHexOrText: string }>;
}

export interface FineTuneControlComparison {
  targetDomain: string;
  evalBytes: number;
  scratchSameBudgetAccPct: number;
  scratchSameBudgetPpl: number;
  reloadedFineTuneAccPct: number;
  reloadedFineTunePpl: number;
  accGainPct: number;
}

export interface NcmUnifiedReport {
  generatedAtIso: string;
  trainTotalBytes: number;
  heldOutTotalBytes: number;
  models: [NcmModelSummary, NcmModelSummary];
  stageBreakdown: NcmStageMetricRow[];
  fineTuneComparison: FineTuneControlComparison;
  honestEvaluationNote: string;
}

export interface X8dResearchSuiteReport {
  generatedAtIso: string;
  totalDurationMs: number;
  codecExperiment: ReversibleCodecReport;
  xAtSequenceExperiment: XAtSequenceReport;
  mp4ExecutionExperiment: Mp4AnalogyReport;
  ncmUnifiedExperiment: NcmUnifiedReport;
  discardedOrFailedHypotheses: Array<{
    hypothesis: string;
    measuredEvidence: string;
    disposition: 'rejected' | 'bounded' | 'needs-larger-scale';
  }>;
  upstreamRecommendations: Array<{
    targetRepo: 'x8Dsub-byte' | 'x8D-Omni-Diffusion';
    recommendation: string;
    evidenceGate: string;
  }>;
}
