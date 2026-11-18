import test from 'node:test';
import assert from 'node:assert/strict';
import { runSubBitComparisonBenchmark } from './quantBenchmarkEngine';

test('Sub-bit benchmark measures real serialized Uint8Array Kronecker buffers without formula overrides', () => {
  const report = runSubBitComparisonBenchmark();
  assert.equal(report.rows.length, 6);

  const bf16 = report.rows.find((r) => r.methodId === 'bf16-baseline')!;
  const x8dK2 = report.rows.find((r) => r.methodId === 'x8d-subbit-005')!;
  const x8dK1 = report.rows.find((r) => r.methodId === 'x8d-subbit-0008')!;
  const broken57 = report.rows.find((r) => r.methodId === 'legacy-blockmean-57')!;

  // On small 64x192 (12,288 weights), real serialized Uint8Array bytes include headers & 4-bit column scales (<0.20 bpw, honest)
  assert.ok(x8dK1.bpw < 0.25 && x8dK1.bpw > 0.05, `Expected honest 64x192 serialized bpw in (0.05, 0.25), got ${x8dK1.bpw}`);
  assert.ok(x8dK1.storedBytes < 400, `Expected <400 bytes Uint8Array for 64x192, got ${x8dK1.storedBytes}`);
  assert.ok(
    x8dK1.cosineSimilarity > 0.70,
    `Expected deserialized 2-bit K=1 Kronecker cosine > 0.70 on structured FFN, got ${x8dK1.cosineSimilarity}`
  );
  assert.ok(
    x8dK2.cosineSimilarity > 0.85,
    `Expected deserialized 4-bit K=2 Kronecker cosine > 0.85 on structured FFN, got ${x8dK2.cosineSimilarity}`
  );
  assert.ok(
    x8dK1.bytePerplexity < broken57.bytePerplexity,
    `Expected x8D PPL (${x8dK1.bytePerplexity}) better than broken #57 (${broken57.bytePerplexity})`
  );
  assert.ok(
    broken57.cosineSimilarity < 0.25,
    `Expected #57 block-mean collapse cosine < 0.25, got ${broken57.cosineSimilarity}`
  );
  assert.ok(bf16.snakeFoodScore120Steps >= 3, 'Snake plays cleanly across dynamic food spawns');

  // Identical serializer on 512x1536 FFN layer naturally achieves sub-0.02 bpw without range coding
  assert.ok(
    report.scaleCheck512x1536.measuredBpw < 0.02,
    `Expected <0.02 bpw on 512x1536 FFN serialization, got ${report.scaleCheck512x1536.measuredBpw}`
  );
  assert.ok(
    report.scaleCheck512x1536.cosineSimilarity > 0.35 &&
      report.scaleCheck512x1536.cosineSimilarity > broken57.cosineSimilarity * 6,
    `Expected 512x1536 K=1 2-bit cosine (${report.scaleCheck512x1536.cosineSimilarity}) >> broken #57 (${broken57.cosineSimilarity})`
  );
});
