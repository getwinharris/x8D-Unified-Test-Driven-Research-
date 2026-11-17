import test from 'node:test';
import assert from 'node:assert/strict';
import { runSubBitComparisonBenchmark } from './quantBenchmarkEngine';

test('Sub-bit benchmark compares BF16, INT8, INT4, x8D Kronecker + x8d_arith (0.008 bpw), and #57 block-mean honestly', () => {
  const report = runSubBitComparisonBenchmark();
  assert.equal(report.rows.length, 6);

  const bf16 = report.rows.find((r) => r.methodId === 'bf16-baseline')!;
  const x8d0008 = report.rows.find((r) => r.methodId === 'x8d-subbit-0008')!;
  const broken57 = report.rows.find((r) => r.methodId === 'legacy-blockmean-57')!;

  assert.ok(x8d0008.bpw <= 0.0085, `Expected <=0.0085 bpw target, got ${x8d0008.bpw}`);
  assert.ok(
    x8d0008.cosineSimilarity > 0.75,
    `Expected structured Kronecker+x8d_arith cosine > 0.75 on structured FFN, got ${x8d0008.cosineSimilarity}`
  );
  assert.ok(
    x8d0008.bytePerplexity < broken57.bytePerplexity,
    `Expected x8D 0.008 bpw PPL (${x8d0008.bytePerplexity}) better than broken #57 (${broken57.bytePerplexity})`
  );
  assert.ok(
    broken57.cosineSimilarity < 0.25,
    `Expected #57 block-mean collapse cosine < 0.25, got ${broken57.cosineSimilarity}`
  );
  assert.ok(bf16.snakeFoodScore120Steps >= 3, 'Snake plays cleanly across dynamic food spawns');
});
