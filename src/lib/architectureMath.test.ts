import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MODEL_SCALE_PRESETS,
  PRECISION_FORMAT_BASELINE,
  classifySubByteTier,
  computeKroneckerSubBit,
  computeParameterBreakdown,
  computeUniqueSubByteArith,
  formatBytes,
  formatParams,
  simulateFootprint,
  simulateUnifiedX8DImage,
} from './architectureMath';

test('Unified .x8D sub-bit binary image packs 50M NCM under 0.002x BF16 with direct-seek tail index', () => {
  const img = simulateUnifiedX8DImage({
    scaleId: 'ncm-50m',
    attentionCodec: 'kron-factored',
    kroneckerTerms: 1,
    kroneckerBits: 4,
    bytePriorAccuracy: 0.999,
  });
  assert.ok(img.unifiedX8DBytes < 160_000, `Expected <160 KB unified .x8D, got ${img.unifiedX8DBytes}`);
  assert.ok(img.compressionFactorVsBf16 > 600, `Expected >600x compression, got ${img.compressionFactorVsBf16}`);
  assert.ok(img.overallBytesPerWeight < 0.003);

  const subbyte0008 = simulateUnifiedX8DImage({
    scaleId: 'ncm-50m',
    attentionCodec: 'kron-subbyte-0008',
    kroneckerTerms: 1,
    kroneckerBits: 4,
    bytePriorAccuracy: 0.999,
  });
  assert.ok(subbyte0008.unifiedX8DBytes < 120_000);
  assert.ok(subbyte0008.compressionFactorVsBf16 > 900);
});

test('Unique per-byte (A=65 -> 0.065) continuous arithmetic coding hits <0.008 bpw without block packing', () => {
  assert.equal(PRECISION_FORMAT_BASELINE.length, 6);
  const res = computeUniqueSubByteArith({
    byteValue: 65,
    streamLengthBytes: 100_000,
    priorMatchRate: 0.9998,
  });
  assert.equal(res.subByteScaledValue, 0.065);
  assert.equal(res.standaloneFixedBits, 8);
  assert.equal(res.preservesUniquePerByte, true);
  assert.ok(res.arithDiskBytes < 100, `Expected <100 bytes on disk, got ${res.arithDiskBytes}`);
  assert.ok(res.arithBitsPerByte < 0.008, `Expected <0.008 bits/byte, got ${res.arithBitsPerByte}`);
});

test('Kronecker sub-byte packing hits the <= 0.001 bytes/weight (1000+ weights/byte) threshold without rank-1 collapse', () => {
  // 512 x 1536 FFN layer with 16x32 outer x 32x48 inner, K=1 @ 2-bit -> 0.00066 B/weight (~1512 weights/byte)
  const ffn = computeKroneckerSubBit({
    rows: 512,
    cols: 1536,
    p1: 16,
    q1: 32,
    kTerms: 1,
    bitsPerFactor: 2,
  });
  assert.ok(ffn.bytesPerElement <= 0.001, `Expected <= 0.001 B/w, got ${ffn.bytesPerElement}`);
  assert.ok(ffn.weightsPerByte >= 1000, `Expected >= 1000 weights/byte, got ${ffn.weightsPerByte}`);
  assert.equal(ffn.subByteTier, '<=0.001');
  assert.ok(ffn.maxAchievableRank >= 256);

  // K=1 @ 3-bit also holds <= 0.001 bytes/weight (0.00099 B/w, ~1013 weights/byte)
  const ffn3Bit = computeKroneckerSubBit({
    rows: 512,
    cols: 1536,
    p1: 16,
    q1: 32,
    kTerms: 1,
    bitsPerFactor: 3,
  });
  assert.ok(ffn3Bit.bytesPerElement <= 0.001, `Expected <= 0.001 B/w, got ${ffn3Bit.bytesPerElement}`);
  assert.equal(classifySubByteTier(0.5), '<=0.5');
  assert.equal(classifySubByteTier(0.25), '<=0.25');
  assert.equal(classifySubByteTier(0.1), '<=0.1');
  assert.equal(classifySubByteTier(0.001), '<=0.001');
});

test('50M Hybrid Omni Diffusion NCM preset stays calibrated around ~50M parameters', () => {
  const breakdown = computeParameterBreakdown(MODEL_SCALE_PRESETS['ncm-50m']);
  assert.ok(
    breakdown.totalParams >= 48_000_000 && breakdown.totalParams <= 53_000_000,
    `Expected ~50M params, got ${breakdown.totalParams}`
  );
  assert.equal(breakdown.byteEmbeddingParams, 256 * 512);
});

test('1B x8D preset matches ~1.03B parameter budget from Issue #68', () => {
  const breakdown = computeParameterBreakdown(MODEL_SCALE_PRESETS['x8d-1b']);
  assert.ok(
    breakdown.totalParams >= 1_000_000_000 && breakdown.totalParams <= 1_120_000_000,
    `Expected ~1.03B-1.10B params, got ${breakdown.totalParams}`
  );
});

test('0.001 block-mean regime flags non-viable forward pass per Issues #57/#59', () => {
  const res = simulateFootprint({
    scaleId: 'ncm-50m',
    quantId: 'block-mean-0001',
    executionMode: 'mmap-latent-file',
    contextBytes: 16384,
    activeLayerSpanRatio: 0.15,
    distributedNodes: 4,
    ncmCoarseBypassRate: 0.65,
  });
  assert.equal(res.forwardPassViable, false);
  assert.ok(res.warnings.some((w) => w.includes('Issues #57/#59')));
  assert.ok(res.diskBytes < 100_000); // ~50.9 KB due to 1000:1 collapse
});

test('U8 x 0.001 raw quanta occupies 8.0 bits/weight (no compression) per Issues #50/#58', () => {
  const res = simulateFootprint({
    scaleId: 'ncm-50m',
    quantId: 'u8-quanta-0001',
    executionMode: 'mmap-latent-file',
    contextBytes: 16384,
    activeLayerSpanRatio: 0.15,
    distributedNodes: 1,
    ncmCoarseBypassRate: 0.5,
  });
  assert.equal(res.effectiveBitsPerWeight, 8);
  assert.equal(res.diskBytes, res.params.totalParams);
  assert.ok(res.warnings.some((w) => w.includes('Issues #50/#58')));
});

test('0.008 bpw Kronecker + x8d_arith shrinks disk and mmap RSS while staying viable', () => {
  const fullFp16 = simulateFootprint({
    scaleId: 'x8d-1b',
    quantId: 'fp16',
    executionMode: 'full-ram-resident',
    contextBytes: 32768,
    activeLayerSpanRatio: 0.15,
    distributedNodes: 8,
    ncmCoarseBypassRate: 0.6,
  });

  const mmapSubByte = simulateFootprint({
    scaleId: 'x8d-1b',
    quantId: 'x8d-subbyte-0008',
    executionMode: 'mmap-latent-file',
    contextBytes: 32768,
    activeLayerSpanRatio: 0.15,
    distributedNodes: 8,
    ncmCoarseBypassRate: 0.6,
  });

  assert.equal(mmapSubByte.forwardPassViable, true);
  assert.equal(mmapSubByte.effectiveBitsPerWeight, 0.008);
  assert.ok(mmapSubByte.compressionRatioVsFp16 >= 1999);
  assert.ok(mmapSubByte.activeWorkingSetRamBytes < fullFp16.activeWorkingSetRamBytes * 0.05);
  assert.ok(mmapSubByte.ncmEffectiveComputeFactor < 0.5);
});

test('1T Edge AGI preset stays under 1 GB disk and 1 GB active RAM in raw-x8d-direct mode (no mmap)', () => {
  const sim1T = simulateFootprint({
    scaleId: 'x8d-1t',
    quantId: 'x8d-subbyte-0008',
    executionMode: 'raw-x8d-direct',
    contextBytes: 16384,
    activeLayerSpanRatio: 0.15,
    distributedNodes: 1,
    ncmCoarseBypassRate: 0.65,
  });
  assert.ok(
    sim1T.params.totalParams >= 980_000_000_000 && sim1T.params.totalParams <= 1_050_000_000_000,
    `Expected ~1.00T params, got ${sim1T.params.totalParams}`
  );
  assert.ok(sim1T.diskBytes < 1024 * 1024 * 1024, `Expected <1 GiB disk, got ${sim1T.diskBytes}`);
  assert.ok(
    sim1T.activeWorkingSetRamBytes < 1024 * 1024 * 1024,
    `Expected <1 GiB active RAM on phone/CPU, got ${sim1T.activeWorkingSetRamBytes}`
  );
  assert.ok(sim1T.warnings.some((w) => w.includes('No mmap')));
});

test('formatting helpers render human-readable units cleanly', () => {
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(1024 * 1024 * 28), '28.00 MB');
  assert.equal(formatParams(50_895_872), '50.90M');
  assert.equal(formatParams(1_005_000_000_000), '1.00T');
});
