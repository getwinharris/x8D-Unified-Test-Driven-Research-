import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SYSTEM_COMPARISON_MATRIX,
  TASK_PRESET_SAMPLES,
  encodeDecodeCodec01,
  simulateUnifiedLoop,
  toX8dQuantaString,
} from './unifiedX8dLoop';

test('x8D native quanta maps every raw byte 0..255 uniquely into 0.000..0.255', () => {
  assert.equal(toX8dQuantaString(65), '0.065');
  assert.equal(toX8dQuantaString(0), '0.000');
  assert.equal(toX8dQuantaString(255), '0.255');
});

test('Codec 0x01 delta+run-length verifies 100% bit-exact lossless roundtrip', () => {
  const sample = new TextEncoder().encode(
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA{"action":[0.065,0.128]}'
  );
  const res = encodeDecodeCodec01(sample);
  assert.equal(res.exactMatch, true);
  assert.ok(res.packedBytes < sample.length);
});

test('Unified x8D NCM-Diffusion loop routes System 1 fast reflex faster than System 2 reasoning', () => {
  assert.equal(SYSTEM_COMPARISON_MATRIX.length, 4);
  const fastGame = simulateUnifiedLoop(TASK_PRESET_SAMPLES['game-reflex'].sample, {
    taskMode: 'game-reflex',
    system1EntropyThreshold: 0.72,
    maxDiffusionIterations: 16,
    ncmCoarseLevels: 3,
  });
  const hardMath = simulateUnifiedLoop(
    TASK_PRESET_SAMPLES['reasoning-math'].sample +
      ' x9Q!z7$Km#2vLp@8Rw&4Bn*1Tc^5Hy(0Js)3Wd+6Fg=8Yx' +
      ' 9182736450qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM',
    {
      taskMode: 'reasoning-math',
      system1EntropyThreshold: 0.4,
      maxDiffusionIterations: 24,
      ncmCoarseLevels: 3,
    }
  );

  assert.equal(fastGame.losslessRoundtripVerified, true);
  assert.equal(hardMath.losslessRoundtripVerified, true);
  assert.ok(fastGame.avgIterationsPerBlock < hardMath.avgIterationsPerBlock);
  assert.ok(fastGame.speedupVsFullDiffusion > 5);

  // Verify fixed 8x8 is only one baseline among 4 internal representations
  for (const rep of ['adaptive-entropy-chunks', 'ncm-concept-hierarchy', 'kda-recurrent-state', 'fixed-8x8-baseline'] as const) {
    const out = simulateUnifiedLoop(TASK_PRESET_SAMPLES['omni-multimodal'].sample, {
      taskMode: 'omni-multimodal',
      representationMode: rep,
      system1EntropyThreshold: 0.65,
      maxDiffusionIterations: 16,
      ncmCoarseLevels: 3,
    });
    assert.equal(out.losslessRoundtripVerified, true);
    assert.ok(out.totalBlocks >= 1);
  }
});
