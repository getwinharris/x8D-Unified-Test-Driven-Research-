import { runCompleteX8dResearchSuite } from '../src/lib/x8dResearchSuiteRunner';

const report = runCompleteX8dResearchSuite();

console.log('\n=== x8D Unified Omni-Diffusion Research Brief Suite ===');
console.log(`Generated: ${report.generatedAtIso} (${report.totalDurationMs} ms)`);
console.log('\n[Section 3] Reversible Codec (8 Payloads, SHA-256 & CRC32):');
for (const row of report.codecExperiment.payloadAudits) {
  console.log(
    ` - ${row.payloadId.padEnd(22)} raw=${String(row.rawBytes).padStart(6)} B -> .x8D=${String(row.physicalX8dBytes).padStart(6)} B | ` +
      `${row.storedBitsPerByte.toFixed(4)} bpb | ratio=${row.compressionRatio.toFixed(3)}x | sha256=${row.sha256ExactMatch ? 'OK' : 'FAIL'}`
  );
}

console.log('\n[Section 5] xAt = 65,536 Byte-State Sequence Comparison:');
for (const row of report.xAtSequenceExperiment.rows) {
  console.log(
    ` - ${row.methodId.padEnd(24)} active=${row.activeRatioPct.toFixed(1)}% (${row.activeBytesEvaluated} B) | ` +
      `needle=${row.needleRetrievalAccPct.toFixed(1)}% | meta=${row.indexMetadataBytes} B`
  );
}

console.log('\n[Section 6] MP4 Direct-Factor Execution (4 Experts x 512x1536):');
for (const row of report.mp4ExecutionExperiment.rows) {
  console.log(
    ` - ${row.modeId.padEnd(26)} buf=${String(row.peakWorkingBufferBytes).padStart(8)} B | ` +
      `diskRead=${String(row.bytesReadFromDisk).padStart(8)} B | maxDiff=${row.maxAbsDiffVsFactorDense} | ${row.latencyMs} ms`
  );
}

console.log('\n[Sections 4, 7 & 8] NCM RVQ Ablation & Stage 1-5 Multimodal Reload-Only Eval:');
for (const m of report.ncmUnifiedExperiment.models) {
  console.log(
    ` - ${m.modelId.padEnd(24)} disk=${m.physicalX8dBytes} B (cb=${m.codebookBytes} B) | ` +
      `PPL=${m.overallHeldOutPpl} | Acc=${m.overallTop1AccPct}% | avgSteps=${m.avgStepsPerSpan}`
  );
}
console.log(
  ` - Stage 5 Fine-Tune Control: Scratch=${report.ncmUnifiedExperiment.fineTuneComparison.scratchSameBudgetAccPct}% -> Reloaded+FT=${report.ncmUnifiedExperiment.fineTuneComparison.reloadedFineTuneAccPct}%`
);
