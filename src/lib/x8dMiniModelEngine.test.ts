import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SUDOKU_PRESETS,
  compareInputEncodings,
  createInitialSnakeState,
  decodeQuantaToText,
  evaluateUnifiedPrompt,
  preTokenizeBpeChunks,
  solveSudokuUnified,
  stepSnakeUnifiedDecision,
  trainMiniX8dModel,
} from './x8dMiniModelEngine.ts';

test('GPT-2 BPE pre_tokenize vs Meta BLT (0..255) vs x8D native (0.000..0.255) comparison', () => {
  const text = "I've solved 9x9 Sudoku!";
  const chunks = preTokenizeBpeChunks(text);
  assert.ok(chunks.includes("'ve"), 'BPE pre_tokenize splits contractions accurately');

  const cmp = compareInputEncodings(text);
  assert.equal(cmp.bltIntIds[0], 73); // 'I' -> 73 in 0..255
  assert.equal(cmp.x8dQuantaIds[0], '0.073'); // 'I' -> 0.073 in 0.000..0.255
  assert.equal(cmp.bpeVocabTableSize, 50257);
  assert.equal(cmp.bltIntIds.length, cmp.x8dQuantaIds.length);
});

test('Mini unified x8D model training decreases loss cleanly at 0.008 bpw', () => {
  const trained = trainMiniX8dModel(12);
  assert.equal(trained.trainedEpochs, 12);
  assert.equal(trained.lossHistory.length, 12);
  assert.ok(
    trained.lossHistory[trained.lossHistory.length - 1].loss < trained.lossHistory[0].loss,
    'Gradient descent reduces training loss over epochs'
  );
  // Safe wall avoidance weight learns negative penalty for danger=0.255
  assert.ok(trained.wSnake[0] < 0, 'Collision danger weight converges negative');
  assert.ok(trained.wSnake[1] > 0, 'Food alignment weight converges positive');
});

test('Trained unified model solves 9x9 Sudoku puzzles via NCM + Diffusion loop with 0 conflicts', () => {
  const trained = trainMiniX8dModel(10);
  for (const preset of SUDOKU_PRESETS) {
    const res = solveSudokuUnified(preset.puzzle81, trained);
    assert.equal(res.solved, true, `Failed solving preset ${preset.id}`);
    assert.equal(res.conflicts, 0);
    assert.equal(res.givensPreserved, true);
    assert.equal(res.solution81.length, 81);
    assert.ok(res.inputQuantaSample.length === 9);
  }
});

test('Same trained model plays Snake decision-by-decision at System 1 Iter 1 with dynamic food spawns', () => {
  const trained = trainMiniX8dModel(12);
  let state = createInitialSnakeState(10, 424242);
  const altReset = createInitialSnakeState(10, 919191);
  assert.notEqual(
    `${state.food.x},${state.food.y}`,
    `${altReset.food.x},${altReset.food.y}`,
    'Distinct seeds spawn food in different starting coordinates'
  );

  let steps = 0;
  let maxScore = 0;
  while (state.alive && steps < 90) {
    state = stepSnakeUnifiedDecision(state, trained);
    assert.ok(state.lastDecision !== null);
    assert.equal(state.lastDecision.routedSystem, 'system-1');
    assert.match(state.lastDecision.actionQuanta, /^0\.0(68|76|82|85)$/);
    maxScore = Math.max(maxScore, state.score);
    steps++;
  }
  assert.ok(state.alive, 'Trained unified model survives 90 Snake decision steps cleanly without deadlocks');
  assert.ok(maxScore >= 3, `Expected snake to eat >=3 dynamic foods, got score=${maxScore}`);
  assert.ok(state.foodSpawnHistory.length >= 4, 'Food spawns at new positions after each eat');
});

test('Prompt evaluator routes deterministic math, multi-clause GR00T, 1T-in-1GB RAM, and decodes 0.000..0.255 back to UTF-8', () => {
  const cmp = compareInputEncodings('Hello x8D!');
  assert.equal(decodeQuantaToText(cmp.x8dQuantaIds), 'Hello x8D!');

  const trained = trainMiniX8dModel(8);
  const mathRes = evaluateUnifiedPrompt('1024 * 64', trained);
  assert.equal(mathRes.routedSystem, 'system-1');
  assert.equal(mathRes.iterationsUsed, 1);
  assert.ok(mathRes.decodedLanguageText.includes('65,536'));

  const multiClauseRes = evaluateUnifiedPrompt(
    'Calculate 1024 * 64 in System 1 or ask about Sudoku what is groot',
    trained
  );
  assert.ok(multiClauseRes.decodedLanguageText.includes('GR00T'));
  assert.ok(multiClauseRes.decodedLanguageText.includes('65,536'));

  const oneTRes = evaluateUnifiedPrompt('How does a 1T model fit in 1GB RAM on a phone?', trained);
  assert.ok(oneTRes.decodedLanguageText.includes('1 Trillion'));
  assert.ok(oneTRes.decodedLanguageText.includes('< 1 GB'));

  const sudoRes = evaluateUnifiedPrompt(`Solve sudoku ${SUDOKU_PRESETS[0].puzzle81}`, trained);
  assert.equal(sudoRes.routedSystem, 'system-2');
  assert.ok(sudoRes.decodedLanguageText.includes('0 conflicts'));
});
