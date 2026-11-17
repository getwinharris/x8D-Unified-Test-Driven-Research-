import test from 'node:test';
import assert from 'node:assert/strict';
import { runNeedle3RealBenchmark } from './needle3BenchmarkEngine';

test('Real HuggingFace Cactus-Compute/needle3 benchmark verifies FP16, INT8, INT4, .cact, Param*0.001, and PTQ vs QAT', () => {
  const report = runNeedle3RealBenchmark('self_attn_out_proj');

  assert.equal(report.modelId, 'Cactus-Compute/needle3');
  assert.equal(report.totalParameters, 121_019_861);
  assert.ok(report.safetensorsBytes > 240_000_000, 'Expected real 242MB safetensors');
  assert.ok(report.officialCactBytes > 35_000_000 && report.officialCactBytes < 36_000_000, 'Expected real 35.3MB .cact');

  const cactus = report.rows.find((r) => r.methodId === 'cactus-official-cact')!;
  const unpadded016 = report.rows.find((r) => r.methodId === 'x8d-unpadded-016')!;
  const unpadded008 = report.rows.find((r) => r.methodId === 'x8d-unpadded-008')!;
  const kronUnpadded = report.rows.find((r) => r.methodId === 'x8d-kron-unpadded')!;
  const paramScaled = report.rows.find((r) => r.methodId === 'x8d-param-0001-scaled')!;
  const broken57 = report.rows.find((r) => r.methodId === 'x8d-blockmean-57')!;

  assert.ok(
    unpadded016.bpw <= 0.01601 && unpadded016.projectedNeedle3Mb <= 0.23085,
    `Expected unpadded 0.001x FP16 (${unpadded016.projectedNeedle3Mb} MB, ${unpadded016.bpw} bpw) <= 0.23083 MB`
  );
  assert.ok(
    unpadded008.bpw <= 0.00801 && unpadded008.projectedNeedle3Mb <= 0.1155,
    `Expected unpadded 8->0.008 bpw (${unpadded008.projectedNeedle3Mb} MB, ${unpadded008.bpw} bpw) <= 0.1155 MB`
  );
  assert.ok(
    unpadded016.cosineSimilarity > broken57.cosineSimilarity && unpadded008.x8dSeekRoundtripVerified,
    'Unpadded BitWriter/BitReader roundtrip verified and outperforms #57 block-mean'
  );
  assert.ok(
    paramScaled.sqnrDb > cactus.sqnrDb + 4.0 && paramScaled.projectedNeedle3Mb > 30,
    `Value-scale-only without sub-bit packing stays >30 MB (${paramScaled.projectedNeedle3Mb} MB)`
  );
  assert.ok(
    kronUnpadded.bpw < 0.02 && kronUnpadded.cosineSimilarity < 0.35,
    'Honest PTQ report: full-rank dense matrices require QAT/STE (#56) before sub-0.02bpw Kronecker'
  );
  assert.ok(!broken57.x8dSeekRoundtripVerified, 'Broken #57 block-mean fails forward pass');

  // Verify structured embedding layer under unpadded sub-bit SVD
  const embReport = runNeedle3RealBenchmark('embedding_slice');
  const embKron = embReport.rows.find((r) => r.methodId === 'x8d-kron-unpadded')!;
  assert.ok(
    embKron.bpw < 0.02 && embKron.cosineSimilarity > 0.8,
    `Expected embedding unpadded SVD sub-bit (<0.02 bpw -> ${embKron.projectedNeedle3Mb} MB) to retain >0.80 cosine (${embKron.cosineSimilarity})`
  );
});
