import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CLOUD_MODEL_TIERS,
  runDualContainerAgenticStep,
  simulateMemoryX8DPacking,
} from './memoryX8dEngine';

test('Cloud 10T and 50T model.x8D presets hit sub-byte 0.008 bpw footprints under <1 GB active RAM', () => {
  const t10 = CLOUD_MODEL_TIERS['cloud-10t'];
  const t50 = CLOUD_MODEL_TIERS['cloud-50t'];

  assert.equal(t10.x8dSubByte0008Bytes, 10_000_000_000);
  assert.equal(t50.x8dSubByte0008Bytes, 50_000_000_000);
  assert.ok(t10.activeSpanRamBytes < 1024 * 1024 * 1024);
  assert.ok(t50.activeSpanRamBytes < 1024 * 1024 * 1024);
});

test('memory.x8D packs AGENTS.md, SKILL.md, MCP schemas, and user prefs with lossless 0.000..0.255 quanta', () => {
  const res = simulateMemoryX8DPacking(
    ['skills-agents-md', 'mcp-schemas', 'files-rw-images', 'user-likes-prefs', 'verified-tool-steps'],
    true
  );

  assert.equal(res.selectedCount, 5);
  assert.equal(res.codecExactVerified, true);
  assert.ok(res.compressionRatio > 15, `Expected >15x memory compression, got ${res.compressionRatio}`);
  assert.ok(res.packedMemoryX8DBytes < res.rawArtifactsBytes * 0.08);
  assert.ok(res.markdownTokensSavedPerTurn > 200_000);
  assert.equal(res.outlierSqnrGainDb, 8.4);
  assert.ok(res.sampleQuanta.every((q) => Number(q) >= 0 && Number(q) <= 0.255));
});

test('Privacy gate locks PII/user files strictly inside local memory.x8D while allowing anonymized skill breakthroughs', () => {
  const privateTask = runDualContainerAgenticStep('personal-workspace-qa', 'cloud-50t', true);
  assert.equal(privateTask.containsPiiOrUserFiles, true);
  assert.equal(privateTask.privacyGateStatus, 'LOCAL_ONLY_LOCKED');

  const skillBreakthrough = runDualContainerAgenticStep('mcp-code-fix', 'cloud-10t', true);
  assert.equal(skillBreakthrough.containsPiiOrUserFiles, false);
  assert.equal(skillBreakthrough.privacyGateStatus, 'ANONYMIZED_BREAKTHROUGH_ELIGIBLE');
  assert.ok(skillBreakthrough.promptMarkdownAvoidedTokens > 40_000);
});
