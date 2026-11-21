import fs from 'fs';
import { auditAndDecontaminateDiskCorpus, readOmniManifest } from '../src/lib/omniDatasetIngestor';
import { OmniUnifiedTrainer, inferFromDiskX8dOnly } from '../src/lib/omniUnifiedTrainer';

const steps = Number(process.env.OMNI_STEPS ?? 240);
const batchSize = Number(process.env.OMNI_BATCH ?? 16);

let manifest = auditAndDecontaminateDiskCorpus() ?? readOmniManifest();
if (!manifest) {
  console.error('No omni datasets on disk yet. Run: npm run fetch:omni');
  process.exit(1);
}

console.log(
  `[Decontam #69] Clean Train=${(manifest.totalTrainBytes / 1024).toFixed(1)} KB | Held-Out=${(manifest.totalHeldOutBytes / 1024).toFixed(1)} KB | Dups Removed=${manifest.dedupStats?.exactDuplicatesRemoved ?? 0} | Overlap Removed=${manifest.dedupStats?.heldOutContaminatedRemoved ?? 0}`
);

const trainer = new OmniUnifiedTrainer();
console.log(
  `Training Unified 5.00M 256-State Diffusion MoE (.x8D v2 STE) (${steps} steps × ${batchSize} windows)...`
);
const metrics = trainer.trainSynchronously(steps, batchSize);

console.log('\n==============================================================');
console.log(`Step ${metrics.step} | Loss ${metrics.initialTrainLoss} -> ${metrics.currentTrainLoss} | Elapsed ${(metrics.elapsedMs / 1000).toFixed(1)}s`);
console.log(`Effective Weights: ${metrics.effectiveWeights.toLocaleString()} | Independent DOFs: ${metrics.trainableDofs.toLocaleString()}`);
console.log(`.x8D v2 File on Disk: ${metrics.checkpointBytesOnDisk.toLocaleString()} B (${metrics.achievedBpw} bpw effective | ${metrics.achievedDofBpb} b/DOF vs ${metrics.fp16EquivalentBytes.toLocaleString()} B FP16)`);
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
  console.log(`\n[f.seek/f.read .x8D v2 Infer] Prompt=${JSON.stringify(p)}`);
  console.log(`  Seek Reads: ${res.diskSeeksPerformed} seeks, ${res.diskBytesSeekRead} B / ${res.diskFileSizeBytes} B | CRC Verified=${res.crcVerifiedCount} | S1=${res.system1EarlyExits} S2=${res.system2Refinements} | Experts=[${res.activeExpertsUsed.join(',')}]`);
  console.log(`  Continuation: ${JSON.stringify(res.continuation)}`);
}

if (!fs.existsSync('checkpoints/omni_unified.x8d')) {
  process.exit(1);
}
