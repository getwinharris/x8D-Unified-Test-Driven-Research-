import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { OmniUnifiedTrainer, inferFromDiskX8dOnly } from './omniUnifiedTrainer';
import { OmniDatasetManifest, OmniDomainId } from './omniDatasetIngestor';

function createTempCorpus(): { dir: string; cleanup: () => void } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'x8d-omni-test-'));
  const domains: OmniDomainId[] = [
    'groot_sim',
    'lerobot_control',
    'tinystories_prose',
    'gsm8k_math',
    'humaneval_code',
    'sudoku_extreme',
  ];

  const samples: Record<OmniDomainId, string> = {
    groot_sim: 'GR00T|env=bimanual_panda_gripper.Threading|ep=0|t=0.50|s=[0.10,0.20]|a=[0.30,0.40]|r=1.0\n'.repeat(18),
    lerobot_control: 'LEROBOT|ds=pusht|ep=1|s=[0.12,0.44]|a=[0.05,-0.10]|r=0.50\n'.repeat(18),
    tinystories_prose: 'TEXT|Once upon a time a tiny robot stacked blocks carefully on a wooden table.\n'.repeat(18),
    gsm8k_math: 'MATH|Q:What is 12 + 30?\nA:12 + 30 = 42. #### 42\n'.repeat(18),
    humaneval_code: 'CODE|HumanEval/0\ndef add(a, b):\n    return a + b\n'.repeat(18),
    sudoku_extreme: 'SUDOKU|src|q=530070000600195000098000060800060003400803001|a=534678912672195348\n'.repeat(18),
  };

  const manifestDomains = {} as OmniDatasetManifest['domains'];
  let totalTrainBytes = 0;
  let totalHeldOutBytes = 0;

  for (const id of domains) {
    const domDir = path.join(dir, id);
    fs.mkdirSync(domDir, { recursive: true });
    const raw = new TextEncoder().encode(samples[id]);
    const split = Math.floor(raw.length * 0.8);
    const tr = raw.slice(0, split);
    const ho = raw.slice(split);
    fs.writeFileSync(path.join(domDir, 'train.bin'), tr);
    fs.writeFileSync(path.join(domDir, 'heldout.bin'), ho);
    totalTrainBytes += tr.length;
    totalHeldOutBytes += ho.length;
    manifestDomains[id] = {
      id,
      label: id,
      sourceRepo: `test/${id}`,
      modality: 'test',
      recordsTrain: 14,
      recordsHeldOut: 4,
      trainBytes: tr.length,
      heldOutBytes: ho.length,
      shardsDownloaded: 1,
      lastSamplePreview: samples[id].slice(0, 60),
    };
  }

  const manifest: OmniDatasetManifest = {
    updatedAtIso: new Date().toISOString(),
    totalTrainBytes,
    totalHeldOutBytes,
    totalRecords: 108,
    domains: manifestDomains,
  };
  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest));

  return {
    dir,
    cleanup: () => fs.rmSync(dir, { recursive: true, force: true }),
  };
}

test('OmniUnifiedTrainer trains natively on raw 0..255 bytes, saves magic-free .x8D, and serves direct fs.readSync inference', () => {
  const { dir, cleanup } = createTempCorpus();
  const ckptPath = path.join(dir, 'test_omni.x8d');
  try {
    const trainer = new OmniUnifiedTrainer({ seed: 42, dataDir: dir, checkpointPath: ckptPath });
    const metrics = trainer.trainSynchronously(40, 12);

    assert.equal(metrics.status, 'completed');
    assert.equal(metrics.step, 40);
    assert.ok(metrics.currentTrainLoss < metrics.initialTrainLoss, 'Expected loss to decrease during compact STE diffusion training');
    assert.ok(fs.existsSync(ckptPath), 'Expected .x8D v2 checkpoint file on disk');
    assert.ok(fs.existsSync(`${ckptPath}.meta.json`), 'Expected .x8D metadata sidecar on disk');
    assert.ok(metrics.checkpointBytesOnDisk > 2000 && metrics.checkpointBytesOnDisk < 14000, `Unexpected size ${metrics.checkpointBytesOnDisk}`);
    assert.ok(metrics.achievedBpw < 0.02, `Expected ultra-low effective bpw < 0.02, got ${metrics.achievedBpw}`);

    // Confirm cold-start reloading from saved .x8D restores step & parameters
    const resumed = new OmniUnifiedTrainer({ seed: 99, dataDir: dir, checkpointPath: ckptPath });
    assert.equal(resumed.metrics.step, 40);
    assert.equal(resumed.metrics.checkpointBytesOnDisk, metrics.checkpointBytesOnDisk);

    // Strictly direct f.seek/f.read inference with zero mmap & verified CRC32
    const infer = inferFromDiskX8dOnly('GR00T|env=', 24, ckptPath);
    assert.equal(infer.diskFileSizeBytes, metrics.checkpointBytesOnDisk);
    assert.ok(infer.diskSeeksPerformed >= 7, `Expected >= 7 f.seek/read operations, got ${infer.diskSeeksPerformed}`);
    assert.ok(infer.crcVerifiedCount >= 5, `Expected >= 5 CRC32-verified tensor spans, got ${infer.crcVerifiedCount}`);
    assert.equal(infer.system1EarlyExits + infer.system2Refinements, 24);
    assert.ok(infer.diskBytesSeekRead > 0 && infer.diskBytesSeekRead <= metrics.checkpointBytesOnDisk);
    assert.equal(infer.outputQuantaSample.length, 10);
    assert.ok(infer.activeExpertsUsed.length >= 1);
  } finally {
    cleanup();
  }
});
