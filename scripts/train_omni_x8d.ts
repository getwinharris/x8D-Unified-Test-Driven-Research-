import fs from 'fs';
import { readOmniManifest } from '../src/lib/omniDatasetIngestor';
import { OmniUnifiedTrainer, inferFromDiskX8dOnly } from '../src/lib/omniUnifiedTrainer';

const steps = Number(process.env.OMNI_STEPS ?? 240);
const batchSize = Number(process.env.OMNI_BATCH ?? 24);

const manifest = readOmniManifest();
if (!manifest) {
  console.error('No omni datasets on disk yet. Run: npm run fetch:omni');
  process.exit(1);
}

const trainer = new OmniUnifiedTrainer();
console.log(
  `Training Unified 0.73M Kronecker MoE (.x8D STE) on ${(manifest.totalTrainBytes / 1024).toFixed(1)} KB real HF Omni data (${steps} steps × ${batchSize} windows)...`
);
const metrics = trainer.trainSynchronously(steps, batchSize);

console.log('\n==============================================================');
console.log(`Step ${metrics.step} | Loss ${metrics.initialTrainLoss} -> ${metrics.currentTrainLoss} | Elapsed ${(metrics.elapsedMs / 1000).toFixed(1)}s`);
console.log(`Effective Weights: ${metrics.effectiveWeights.toLocaleString()} | Compact DOFs: ${metrics.trainableDofs.toLocaleString()}`);
console.log(`.x8D File on Disk: ${metrics.checkpointBytesOnDisk.toLocaleString()} B (${metrics.achievedBpw} bpw vs ${metrics.fp16EquivalentBytes.toLocaleString()} B FP16)`);
console.log('Per-Domain Held-Out Perplexity (init -> current):');
for (const [dom, stats] of Object.entries(metrics.perDomainHeldOutPpl)) {
  if (!stats) continue;
  console.log(` - ${dom.padEnd(20)}: PPL ${stats.initPpl} -> ${stats.currentPpl} (Top-1 ${stats.top1AccPct}%)`);
}

const prompts = [
  'GR00T|env=bimanual_panda_gripper.Threading|',
  'SUDOKU|puzzles4_forum_hardest_1905|q=',
  'MATH|Q:',
];
for (const p of prompts) {
  const res = inferFromDiskX8dOnly(p, 40);
  console.log(`\n[f.seek/f.read Disk .x8D Infer] Prompt=${JSON.stringify(p)}`);
  console.log(`  Seek Reads: ${res.diskSeeksPerformed} seeks, ${res.diskBytesSeekRead} B / ${res.diskFileSizeBytes} B | Experts=[${res.activeExpertsUsed.join(',')}]`);
  console.log(`  Continuation: ${JSON.stringify(res.continuation)}`);
}

if (!fs.existsSync('checkpoints/omni_unified.x8d')) {
  process.exit(1);
}
