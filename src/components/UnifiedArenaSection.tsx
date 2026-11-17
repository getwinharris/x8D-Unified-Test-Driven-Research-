import React, { useEffect, useMemo, useState } from 'react';
import { Play, RotateCcw, StepForward, Zap } from 'lucide-react';
import {
  MiniX8dUnifiedWeights,
  QaPromptTestResult,
  SUDOKU_PRESETS,
  SnakeGameState,
  SudokuSolveResult,
  compareInputEncodings,
  createInitialSnakeState,
  createUntrainedMiniWeights,
  evaluateUnifiedPrompt,
  solveSudokuUnified,
  stepSnakeUnifiedDecision,
  trainMiniX8dModel,
} from '../lib/x8dMiniModelEngine';

export const UnifiedArenaSection: React.FC = () => {
  // 1. Tokenization comparator state
  const [compareText, setCompareText] = useState<string>(
    "I've solved 9x9 Sudoku and steered Snake in 0.000..0.255 bytes!"
  );
  const encodingCmp = useMemo(() => compareInputEncodings(compareText), [compareText]);

  // 2. Real SGD mini-model weights (defaults to untrained; user can train live)
  const [weights, setWeights] = useState<MiniX8dUnifiedWeights>(() => createUntrainedMiniWeights());

  // 3. Sudoku Solver test state
  const [selectedSudokuId, setSelectedSudokuId] = useState<string>(SUDOKU_PRESETS[0].id);
  const [sudokuResult, setSudokuResult] = useState<SudokuSolveResult | null>(null);
  const activeSudokuPreset =
    SUDOKU_PRESETS.find((p) => p.id === selectedSudokuId) ?? SUDOKU_PRESETS[0];

  // 4. Decision-by-decision Snake Game state
  const [snakeState, setSnakeState] = useState<SnakeGameState>(() => createInitialSnakeState(10));
  const [snakeAutoPlay, setSnakeAutoPlay] = useState<boolean>(false);

  // 5. Question / Prompt Solving test state
  const [qaInput, setQaInput] = useState<string>(
    'What is GR00T and how does a 1T x8D model fit in 1GB RAM?'
  );
  const [qaResult, setQaResult] = useState<QaPromptTestResult | null>(null);

  // Auto-play timer for Snake decision-by-decision ticks
  useEffect(() => {
    if (!snakeAutoPlay || !snakeState.alive) {
      if (snakeAutoPlay && !snakeState.alive) setSnakeAutoPlay(false);
      return;
    }
    const timer = window.setInterval(() => {
      setSnakeState((prev) => stepSnakeUnifiedDecision(prev, weights));
    }, 180);
    return () => window.clearInterval(timer);
  }, [snakeAutoPlay, snakeState.alive, weights]);

  const handleTrainMiniModel = () => {
    const trained = trainMiniX8dModel(12);
    setWeights(trained);
    // Immediately verify Sudoku & Q&A once trained
    setSudokuResult(solveSudokuUnified(activeSudokuPreset.puzzle81, trained));
    setQaResult(evaluateUnifiedPrompt(qaInput, trained));
  };

  const handleSolveSudoku = () => {
    setSudokuResult(solveSudokuUnified(activeSudokuPreset.puzzle81, weights));
  };

  const handleSnakeStep = () => {
    setSnakeAutoPlay(false);
    setSnakeState((prev) => stepSnakeUnifiedDecision(prev, weights));
  };

  const handleSnakeReset = () => {
    setSnakeAutoPlay(false);
    setSnakeState(createInitialSnakeState(10));
  };

  const handleEvaluateQa = (e: React.FormEvent) => {
    e.preventDefault();
    setQaResult(evaluateUnifiedPrompt(qaInput, weights));
  };

  const snakeBodyKeys = useMemo(
    () => new Set(snakeState.snake.slice(1).map((p) => `${p.x},${p.y}`)),
    [snakeState.snake]
  );

  return (
    <section id="unified-arena" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          03. Trainable Micro-Model &amp; Live Sudoku · Snake · Byte Input Arena
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Raw 8-Bit Bytes → 0.000–0.255 Quanta (vs. BPE &amp; BLT) · Train &amp; Test One Model on Sudoku, Snake &amp; Q&amp;A
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Following <span className="font-mono text-slate-800">x8d_quanta.py</span>,{' '}
          <span className="font-mono text-slate-800">Colibrì</span> tiered NVMe design, and{' '}
          <span className="font-mono text-slate-800">NCM (arXiv:2610.14162 / ncm-mlx)</span>, default dataset and
          prompt 8-bit bytes map straight to fractional <span className="font-mono text-slate-800">0.000–0.255</span> IDs
          (rather than 50,257-entry GPT-2 BPE or BLT&apos;s <span className="font-mono text-slate-800">0..255</span> integers).
          Train the miniature unified model below at the <span className="font-mono text-slate-800">0.008 bpw</span> sub-byte
          target and test that same model live across Sudoku, decision-by-decision Snake, and Q&amp;A.
        </p>
      </div>

      {/* 1. BPE vs Meta BLT (0..255) vs ncm-mlx vs x8D (0.000..0.255) Comparator */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-slate-900">
            Input Representation Comparison: TinyStories BPE vs. Meta BLT (0..255) vs. x8D (0.000–0.255)
          </h3>
          <span className="text-xs font-mono text-slate-500">Default 8-bit Prompt / Dataset Input</span>
        </div>

        <div>
          <label htmlFor="tok-compare-input" className="block text-[11px] font-mono uppercase tracking-wider text-slate-500 mb-1.5">
            Live Input Text (Tested Simultaneously Across Tokenizers)
          </label>
          <input
            id="tok-compare-input"
            type="text"
            value={compareText}
            onChange={(e) => setCompareText(e.target.value)}
            className="w-full rounded border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-900 focus:border-blue-600 focus:outline-none"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="rounded border border-slate-200 bg-slate-50 p-3">
            <div className="font-mono text-[11px] font-semibold text-slate-700">1. TinyStories Byte-BPE</div>
            <div className="mt-1 font-mono text-sm font-semibold text-slate-900 tabular-nums">
              {encodingCmp.bpeTokenCount} chunks · {encodingCmp.bpeVocabTableSize.toLocaleString()} vocab
            </div>
            <p className="mt-1 text-[11px] text-slate-600">
              Requires JSON vocab + merge ranks + <span className="font-mono">unicodedata</span> contractions. Breaks/splits on Sudoku grids &amp; robot actions.
            </p>
            <div className="mt-2 font-mono text-[11px] text-slate-700 break-all">
              Chunks: [{encodingCmp.bpeChunks.slice(0, 5).map((c) => JSON.stringify(c)).join(', ')}]
            </div>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3">
            <div className="font-mono text-[11px] font-semibold text-slate-700">2. Meta BLT (Byte Latent)</div>
            <div className="mt-1 font-mono text-sm font-semibold text-slate-900 tabular-nums">
              {encodingCmp.rawByteLength} bytes · IDs 0..255
            </div>
            <p className="mt-1 text-[11px] text-slate-600">
              Uses raw discrete integer IDs <span className="font-mono">0..255</span> + separate local entropy patcher model.
            </p>
            <div className="mt-2 font-mono text-[11px] text-slate-700 break-all">
              BLT IDs: [{encodingCmp.bltIntIds.slice(0, 7).join(', ')}...]
            </div>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3">
            <div className="font-mono text-[11px] font-semibold text-slate-700">3. ncm-mlx Char + RVQ</div>
            <div className="mt-1 font-mono text-sm font-semibold text-slate-900 tabular-nums">
              {encodingCmp.ncmCharVocabSize} alphabet · 2-Stage
            </div>
            <p className="mt-1 text-[11px] text-slate-600">
              Fixed padded chars + separate Stage-1 RVQ Encoder/Decoder before Stage-2 NCM Generator.
            </p>
            <div className="mt-2 font-mono text-[11px] text-slate-700">
              Pad to fixed L · separate VQ codec
            </div>
          </div>

          <div className="rounded border border-blue-300 bg-blue-50/50 p-3">
            <div className="font-mono text-[11px] font-semibold text-blue-800">4. x8D Unified (0.000–0.255)</div>
            <div className="mt-1 font-mono text-sm font-semibold text-blue-900 tabular-nums">
              {encodingCmp.rawByteLength} bytes · 0.000..0.255
            </div>
            <p className="mt-1 text-[11px] text-slate-700">
              Every 8-bit input byte maps directly (<span className="font-mono">byte × 0.001</span>) into one NCM-Diffusion loop. Zero BPE merges.
            </p>
            <div className="mt-2 font-mono text-[11px] text-blue-800 break-all">
              Quanta: [{encodingCmp.x8dQuantaIds.slice(0, 6).join(', ')}...]
            </div>
          </div>
        </div>
      </div>

      {/* 2. Train Mini Portion Control Strip */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold text-slate-900">
              Mini Unified 0.000–0.255 Backbone Training Status
            </span>
            <span
              className={`rounded border px-2 py-0.5 font-mono text-[11px] ${
                weights.trainedEpochs > 0
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                  : 'border-amber-200 bg-amber-50 text-amber-800'
              }`}
            >
              {weights.trainedEpochs > 0
                ? `Trained ${weights.trainedEpochs} Epochs · Loss ${weights.lossHistory[0]?.loss} → ${weights.lossHistory[weights.lossHistory.length - 1]?.loss}`
                : 'Untrained Prior (Click Train to run SGD)'}
            </span>
          </div>
          <p className="text-xs text-slate-600 font-mono">
            Learned Weights (0.008 bpw target, zero -1 ternary): Snake[danger={weights.wSnake[0]}, food={weights.wSnake[1]}, space={weights.wSnake[2]}] · Sudoku[single={weights.wSudoku[2]}]
          </p>
        </div>

        <button
          type="button"
          onClick={handleTrainMiniModel}
          className="inline-flex items-center justify-center gap-2 rounded-md bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-blue-700 transition-colors cursor-pointer shrink-0"
        >
          <Zap className="w-4 h-4" />
          <span>Train Mini Unified Model (12 SGD Epochs)</span>
        </button>
      </div>

      {/* 3. Three Live Test Modes Using THE ONE Model */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Test Mode A: 9x9 Sudoku Solver (System 2 NCM + Diffusion Loop) */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between pb-2.5 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-900">Test A · 9×9 Sudoku (System 2 Loop)</h3>
            <span className="font-mono text-[11px] text-blue-700">0.049..0.057</span>
          </div>

          <div className="flex gap-2">
            <select
              aria-label="Sudoku Preset"
              value={selectedSudokuId}
              onChange={(e) => {
                setSelectedSudokuId(e.target.value);
                setSudokuResult(null);
              }}
              className="flex-1 rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900"
            >
              {SUDOKU_PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={handleSolveSudoku}
              className="rounded bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-700 cursor-pointer shrink-0"
            >
              Solve
            </button>
          </div>

          {/* 9x9 Grid Display */}
          <div className="grid grid-cols-9 border border-slate-400 rounded overflow-hidden font-mono text-xs text-center tabular-nums">
            {Array.from({ length: 81 }, (_, idx) => {
              const r = Math.floor(idx / 9);
              const c = idx % 9;
              const givenChar = activeSudokuPreset.puzzle81[idx];
              const isGiven = givenChar >= '1' && givenChar <= '9';
              const displayChar = sudokuResult ? sudokuResult.solution81[idx] : isGiven ? givenChar : '·';
              const thickRight = c === 2 || c === 5 ? 'border-r-2 border-r-slate-400' : 'border-r border-r-slate-200';
              const thickBottom = r === 2 || r === 5 ? 'border-b-2 border-b-slate-400' : 'border-b border-b-slate-200';
              return (
                <div
                  key={idx}
                  className={`py-1.5 ${thickRight} ${thickBottom} ${
                    isGiven
                      ? 'bg-slate-100 font-bold text-slate-900'
                      : sudokuResult
                        ? 'bg-blue-50/60 font-semibold text-blue-700'
                        : 'bg-white text-slate-400'
                  }`}
                >
                  {displayChar}
                </div>
              );
            })}
          </div>

          {sudokuResult ? (
            <div className="rounded border border-emerald-200 bg-emerald-50/60 p-3 text-xs space-y-1">
              <div className="font-mono font-semibold text-emerald-900">
                {sudokuResult.solved ? '● Solved 0 Conflicts' : 'Unsolved'} · {sudokuResult.iterationsUsed} NCM+Diff Loops
              </div>
              <div className="font-mono text-[11px] text-slate-700 break-all">
                Row-0 Quanta: [{sudokuResult.outputQuantaSample.join(', ')}]
              </div>
            </div>
          ) : (
            <div className="text-[11px] font-mono text-slate-500">
              Click &quot;Solve&quot; (or &quot;Train&quot;) to solve in-loop using 0.049..0.057 cell quanta.
            </div>
          )}
        </div>

        {/* Test Mode B: Decision-by-Decision Snake Game (System 1 Iter-1 Expert) */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between pb-2.5 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-900">Test B · Snake (System 1 Iter 1)</h3>
            <span className="font-mono text-[11px] text-emerald-700">
              Score {snakeState.score} · Tick {snakeState.tick}
            </span>
          </div>

          {/* 10x10 Snake Grid */}
          <div
            className="grid grid-cols-10 gap-0.5 rounded border border-slate-300 bg-slate-200 p-1 mx-auto max-w-[260px]"
            role="img"
            aria-label={`Snake 10 by 10 grid at tick ${snakeState.tick} with score ${snakeState.score}`}
          >
            {Array.from({ length: 100 }, (_, i) => {
              const x = i % 10;
              const y = Math.floor(i / 10);
              const isHead = snakeState.snake[0].x === x && snakeState.snake[0].y === y;
              const isBody = snakeBodyKeys.has(`${x},${y}`);
              const isFood = snakeState.food.x === x && snakeState.food.y === y;
              return (
                <div
                  key={i}
                  className={`aspect-square rounded-[2px] ${
                    isHead
                      ? 'bg-emerald-700'
                      : isBody
                        ? 'bg-emerald-400'
                        : isFood
                          ? 'bg-amber-500'
                          : 'bg-white'
                  }`}
                />
              );
            })}
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleSnakeStep}
              disabled={!snakeState.alive}
              className="flex-1 inline-flex items-center justify-center gap-1.5 rounded bg-emerald-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-800 disabled:opacity-50 cursor-pointer"
            >
              <StepForward className="w-3.5 h-3.5" />
              <span>Step +1 Decision</span>
            </button>
            <button
              type="button"
              onClick={() => setSnakeAutoPlay((v) => !v)}
              disabled={!snakeState.alive}
              className="inline-flex items-center justify-center gap-1 rounded border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-800 hover:bg-slate-100 disabled:opacity-50 cursor-pointer"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{snakeAutoPlay ? 'Pause' : 'Auto'}</span>
            </button>
            <button
              type="button"
              onClick={handleSnakeReset}
              aria-label="Reset Snake Game"
              className="inline-flex items-center justify-center rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3 text-xs font-mono space-y-1">
            {snakeState.lastDecision ? (
              <>
                <div className="text-slate-900 font-semibold">
                  Decision #{snakeState.lastDecision.tick}: {snakeState.lastDecision.action} (byte{' '}
                  {snakeState.lastDecision.actionByte} → {snakeState.lastDecision.actionQuanta}) · S1 ·{' '}
                  {snakeState.lastDecision.latencyMs} ms
                </div>
                <div className="text-[11px] text-slate-600 break-all">
                  Obs Quanta: [{snakeState.lastDecision.inputBytesSampleQuanta.join(', ')}]
                </div>
              </>
            ) : (
              <div className="text-slate-500 text-[11px]">
                Click &quot;Step +1 Decision&quot; to run System 1 Iter-1 move inference using the same weights.
              </div>
            )}
          </div>
        </div>

        {/* Test Mode C: Prompt / Question Solving in 0.000..0.255 */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between pb-2.5 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-900">Test C · Question Solving</h3>
            <span className="font-mono text-[11px] text-slate-500">S1 + S2 Prompt Gate</span>
          </div>

          <form onSubmit={handleEvaluateQa} className="space-y-2.5">
            <textarea
              rows={3}
              aria-label="Prompt or question to evaluate in 0.000..0.255 space"
              value={qaInput}
              onChange={(e) => setQaInput(e.target.value)}
              className="w-full rounded border border-slate-300 bg-white p-2.5 font-mono text-xs text-slate-900 focus:border-blue-600 focus:outline-none"
            />
            <div className="flex flex-wrap gap-1.5">
              {[
                '1024 * 64',
                'What is GR00T?',
                '1T model in 1GB RAM on phone?',
                'Solve sudoku grid',
                'Predict next Snake move',
              ].map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => {
                    setQaInput(q);
                    setQaResult(evaluateUnifiedPrompt(q, weights));
                  }}
                  className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] font-mono text-slate-700 hover:bg-slate-100 cursor-pointer"
                >
                  {q}
                </button>
              ))}
            </div>
            <button
              type="submit"
              className="w-full rounded bg-slate-900 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-800 cursor-pointer"
            >
              Run Prompt in 0.000–0.255 Unified Model
            </button>
          </form>

          {qaResult && (
            <div className="rounded border border-slate-200 bg-slate-50 p-3 text-xs space-y-2" data-testid="qa-result-card">
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono font-semibold text-slate-900">
                  {qaResult.routedSystem === 'system-1' ? 'System 1 (1 Iter)' : `System 2 (${qaResult.iterationsUsed} Iters)`}
                </span>
                <span className="font-mono text-[10px] text-emerald-700">
                  0.000–0.255 → UTF-8 Decoded
                </span>
              </div>
              <div className="text-slate-900 font-medium leading-relaxed" data-testid="qa-decoded-answer">
                {qaResult.decodedLanguageText}
              </div>
              <div className="text-[11px] text-slate-600">{qaResult.rationale}</div>
              <details className="pt-1 border-t border-slate-200 text-[11px] font-mono text-slate-500">
                <summary className="cursor-pointer hover:text-slate-800">
                  Inspect 0.000–0.255 I/O Byte Quanta Trace
                </summary>
                <div className="mt-1.5 space-y-1 text-blue-700 break-all">
                  <div>In Quanta: [{qaResult.inputQuanta.slice(0, 8).join(', ')}...]</div>
                  <div>Out Quanta: [{qaResult.answerQuanta.slice(0, 8).join(', ')}...]</div>
                </div>
              </details>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
