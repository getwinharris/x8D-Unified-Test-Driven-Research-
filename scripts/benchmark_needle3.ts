import { runNeedle3RealBenchmark } from '../src/lib/needle3BenchmarkEngine';
import { NEEDLE3_TENSOR_SPECS, Needle3TensorId } from '../src/lib/needle3Types';

const argTensor = (process.argv[2] as Needle3TensorId | undefined) ?? 'self_attn_out_proj';
const tensorId: Needle3TensorId =
  argTensor in NEEDLE3_TENSOR_SPECS ? argTensor : 'self_attn_out_proj';

const report = runNeedle3RealBenchmark(tensorId);

console.log('================================================================================================================');
console.log(`REAL HUGGINGFACE MODEL BENCHMARK: ${report.modelId} (${report.totalParameters.toLocaleString()} params)`);
console.log(
  `Artifact Sizes: needle3.safetensors=${(report.safetensorsBytes / (1024 * 1024)).toFixed(2)} MB | ` +
    `needle3.cact=${(report.officialCactBytes / (1024 * 1024)).toFixed(2)} MB (${report.officialCactBpw} bpw)`
);
console.log(
  `Active Tensor: ${report.tensor.label} | min=${report.tensorStats.min}, max=${report.tensorStats.max}, rms=${report.tensorStats.rms}, |w|<0.01=${report.tensorStats.nearZeroPct}%`
);
console.log(`Mode: Direct fs.open/read (no mmap) | Elapsed: ${report.benchmarkDurationMs} ms`);
console.log('================================================================================================================');

const header = [
  'Method'.padEnd(44),
  'BPW'.padStart(8),
  'Proj MB'.padStart(8),
  'Cosine'.padStart(8),
  'SQNR dB'.padStart(8),
  'ActCos'.padStart(8),
  'LogitKL'.padStart(8),
].join(' | ');
console.log(header);
console.log('-'.repeat(header.length));

for (const r of report.rows) {
  console.log(
    [
      r.label.padEnd(44),
      r.bpw.toFixed(4).padStart(8),
      (r.projectedNeedle3Mb < 1 ? r.projectedNeedle3Mb.toFixed(4) : r.projectedNeedle3Mb.toFixed(2)).padStart(8),
      r.cosineSimilarity.toFixed(4).padStart(8),
      r.sqnrDb.toFixed(2).padStart(8),
      r.outputActCosine.toFixed(4).padStart(8),
      r.logitKlDivergence.toFixed(4).padStart(8),
    ].join(' | ')
  );
}

console.log('\nArchitect Takeaway:');
console.log(report.architectTakeaway);
