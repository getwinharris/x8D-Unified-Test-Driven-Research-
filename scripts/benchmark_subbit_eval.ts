import { runSubBitComparisonBenchmark } from '../src/lib/quantBenchmarkEngine';

const report = runSubBitComparisonBenchmark();

console.log('====================================================================================================');
console.log(`x8D SIDE-BY-SIDE QUANTIZATION, PERPLEXITY & TASK BENCHMARK (${report.matrixShape})`);
console.log(`Evaluated Weights: ${report.totalWeightsEvaluated.toLocaleString()} | Eval Stream: ${report.evalStreamBytes} bytes`);
console.log('====================================================================================================');

const header = [
  'Method'.padEnd(36),
  'BPW'.padStart(8),
  'Bytes'.padStart(8),
  'Cosine'.padStart(8),
  'PPL'.padStart(8),
  'dPPL'.padStart(8),
  'Top1%'.padStart(7),
  'Snake'.padStart(7),
].join(' | ');
console.log(header);
console.log('-'.repeat(header.length));

for (const r of report.rows) {
  console.log(
    [
      r.label.padEnd(36),
      r.bpw.toFixed(4).padStart(8),
      String(r.storedBytes).padStart(8),
      r.cosineSimilarity.toFixed(4).padStart(8),
      r.bytePerplexity.toFixed(2).padStart(8),
      (r.pplDeltaVsBf16 >= 0 ? `+${r.pplDeltaVsBf16.toFixed(2)}` : r.pplDeltaVsBf16.toFixed(2)).padStart(8),
      `${r.headAccuracyPct.toFixed(1)}%`.padStart(7),
      `${r.snakeFoodScore120Steps} pts`.padStart(7),
    ].join(' | ')
  );
}

console.log('\nStructural Insight:');
console.log(report.shannonWallInsight);
