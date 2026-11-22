import fs from 'fs';
import os from 'os';
import path from 'path';
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildValidSamplePng,
  buildValidSampleWav,
  packDeltaRleMilli,
  runReversibleCodecExperiment,
  testFixedPointVsFloatRoundtrip,
  unpackDeltaRleMilli,
} from './reversibleCodecExperiment';
import { runMp4DirectExecutionBenchmark, runXAtSequenceExperiment } from './xAtSequenceAndMp4Engine';
import { runNcmUnifiedStageExperiment } from './ncmUnifiedStageTrainer';
import { readCachedX8dResearchReport, runCompleteX8dResearchSuite } from './x8dResearchSuiteRunner';

test('Fixed-point milli-units achieve 0/256 collisions while naive Float32 trunc exhibits drift', () => {
  const rows = testFixedPointVsFloatRoundtrip();
  const intRow = rows.find((r) => r.method.includes('Integer Fixed-Point'));
  const f64Row = rows.find((r) => r.method.includes('Float64'));
  const f32TruncRow = rows.find((r) => r.method.includes('Trunc'));

  assert.ok(intRow && f64Row && f32TruncRow);
  assert.equal(intRow.collisionCount, 0);
  assert.equal(intRow.exactBijectionPct, 100);
  assert.equal(f64Row.collisionCount, 0);
  assert.ok(f32TruncRow.collisionCount > 0, 'Unsafe Float32 trunc should catch binary drift on 0.001 steps');
});

test('Delta-RLE milli packer handles empty, constant, and random-byte payloads bit-exactly', () => {
  const sample = new Uint8Array([0, 0, 0, 0, 65, 66, 67, 255, 255, 255, 10, 200]);
  const packed = packDeltaRleMilli(sample);
  const unpacked = unpackDeltaRleMilli(packed, sample.length);
  assert.deepEqual(unpacked, sample);
});

test('Section 3 reversible codec verifies real PNG, real WAV, 0..255, Tamil, high-entropy random & 65536B .x8D roundtrips with SHA-256 parity', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'x8d-codec-test-'));
  try {
    const png = buildValidSamplePng(16, 16);
    assert.equal(png[0], 0x89);
    assert.equal(String.fromCharCode(...png.subarray(1, 4)), 'PNG');

    const wav = buildValidSampleWav(8000, 800);
    assert.equal(String.fromCharCode(...wav.subarray(0, 4)), 'RIFF');
    assert.equal(String.fromCharCode(...wav.subarray(8, 12)), 'WAVE');

    const report = runReversibleCodecExperiment(tmpDir);
    assert.equal(report.allPayloadsBitExact, true);
    assert.equal(report.pngValidHeaderAndChunks, true);
    assert.equal(report.wavValidRiffHeader, true);
    assert.equal(report.payloadAudits.length, 8);

    for (const row of report.payloadAudits) {
      assert.equal(row.sha256ExactMatch, true, `SHA-256 mismatch on ${row.payloadId}`);
      assert.equal(row.crc32Verified, true, `CRC32 failed on ${row.payloadId}`);
      assert.equal(row.reverseMatchPct, 100);
      assert.ok(row.physicalX8dBytes >= row.payloadSpanBytes);
    }

    const randomRow = report.payloadAudits.find((r) => r.payloadId === 'high-entropy-random');
    assert.ok(randomRow && randomRow.storedBitsPerByte > 8.0, 'Random binary must honestly show >= 8 bpb');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('Section 5 xAt=65,536 sequence benchmark separates sequence capacity and measures adaptive vs fixed 8x8', () => {
  const rep = runXAtSequenceExperiment();
  assert.equal(rep.sequenceBytes, 65536);
  assert.equal(rep.rows.length, 4);

  const adaptive = rep.rows.find((r) => r.methodId === 'adaptive-entropy-cache');
  const fixed = rep.rows.find((r) => r.methodId === 'fixed-8x8-legacy');
  assert.ok(adaptive && fixed);
  assert.ok(adaptive.activeRatioPct < 100, 'Adaptive entropy cache should skip repeated/idle spans');
  assert.ok(adaptive.needleRetrievalAccPct >= fixed.needleRetrievalAccPct);
});

test('Section 6 MP4 direct-factor math matches full dense matmul within <1e-4 while eliminating dense buffer', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'x8d-mp4-test-'));
  try {
    const rep = runMp4DirectExecutionBenchmark(tmpDir);
    assert.equal(rep.rows.length, 5);

    const mode1 = rep.rows[0];
    const mode4 = rep.rows[3];
    const mode5 = rep.rows[4];

    assert.equal(mode1.materializedDenseMatrix, true);
    assert.equal(mode4.materializedDenseMatrix, false);
    assert.equal(mode5.materializedDenseMatrix, false);
    assert.ok(mode4.maxAbsDiffVsFactorDense < 1e-4, `Unexpected algebra drift: ${mode4.maxAbsDiffVsFactorDense}`);
    assert.ok(mode5.cosineVsFactorDense > 0.99999);
    assert.ok(mode5.peakWorkingBufferBytes < mode1.peakWorkingBufferBytes / 100);
    assert.ok(mode5.bytesReadFromDisk < mode1.bytesReadFromDisk / 50);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('Sections 4 & 7-8 NCM RVQ ablation & Stage 1-5 multimodal reload-only eval run cleanly from .x8D disk checkpoint', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'x8d-ncm-test-'));
  try {
    const rep = runNcmUnifiedStageExperiment(tmpDir);
    assert.equal(rep.models.length, 2);
    assert.equal(rep.stageBreakdown.length, 5);

    const flat = rep.models[0];
    const ncm = rep.models[1];
    assert.equal(flat.reloadParityMaxDiff, 0);
    assert.equal(ncm.reloadParityMaxDiff, 0);
    assert.ok(ncm.codebookBytes > 0);
    assert.ok(ncm.overallHeldOutPpl < 256, 'Trained reloaded NCM model should beat random 256 PPL');
    assert.ok(ncm.avgStepsPerSpan <= flat.avgStepsPerSpan);

    const suite = runCompleteX8dResearchSuite(tmpDir);
    const cached = readCachedX8dResearchReport(tmpDir);
    assert.ok(cached);
    assert.equal(cached.codecExperiment.allPayloadsBitExact, suite.codecExperiment.allPayloadsBitExact);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
