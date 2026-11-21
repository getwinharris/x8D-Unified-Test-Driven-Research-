import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Cpu, Database, Download, HardDrive, Play, RotateCcw, Square, StepForward, Zap } from 'lucide-react';
import type { OmniDatasetManifest, OmniDomainId } from '../lib/omniDatasetIngestor';
import type { OmniDiskInferResult, OmniTrainingMetricsSnapshot } from '../lib/omniUnifiedTrainer';
import type { X8dContainerDirectory } from '../lib/x8dContainerFormat';
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
import {
  NativeCompactExperimentReport,
  runNativeCompactTrainingExperiment,
} from '../lib/nativeCompactTrainEngine';

function formatCodecLabel(codecId: number): string {
  switch (codecId) {
    case 1:
      return 'raw_quanta_u8 (0x01)';
    case 2:
      return 'kron_q2+arith (0x02)';
    case 3:
      return 'kron_q4 (0x03)';
    case 4:
      return 'kron_arith_q3 (0x04)';
    default:
      return `codec_${codecId}`;
  }
}

export const UnifiedArenaSection: React.FC = () => {
  const [compareText, setCompareText] = useState<string>(
    "I've solved 9x9 Sudoku and steered Snake in 0.000..0.255 bytes!"
  );
  const encodingCmp = useMemo(() => compareInputEncodings(compareText), [compareText]);

  const [weights, setWeights] = useState<MiniX8dUnifiedWeights>(() => createUntrainedMiniWeights());
  const [selectedSudokuId, setSelectedSudokuId] = useState<string>(SUDOKU_PRESETS[0].id);
  const [sudokuResult, setSudokuResult] = useState<SudokuSolveResult | null>(null);
  const activeSudokuPreset =
    SUDOKU_PRESETS.find((p) => p.id === selectedSudokuId) ?? SUDOKU_PRESETS[0];

  const [snakeState, setSnakeState] = useState<SnakeGameState>(() => createInitialSnakeState(10));
  const [snakeAutoPlay, setSnakeAutoPlay] = useState<boolean>(false);

  const [qaInput, setQaInput] = useState<string>(
    'What is GR00T and how does a 1T x8D model fit in 1GB RAM?'
  );
  const [qaResult, setQaResult] = useState<QaPromptTestResult | null>(null);

  const [nativeConfigId, setNativeConfigId] = useState<'k4-b4' | 'k2-b4' | 'k2-b2'>('k4-b4');
  const [nativeReport, setNativeReport] = useState<NativeCompactExperimentReport | null>(null);
  const [isTrainingNative, setIsTrainingNative] = useState<boolean>(false);

  // Live Full-Corpus Server Omni Training & Strict fs.readSync .x8D Runtime State
  const [omniManifest, setOmniManifest] = useState<OmniDatasetManifest | null>(null);
  const [omniTraining, setOmniTraining] = useState<OmniTrainingMetricsSnapshot | null>(null);
  const [omniContainer, setOmniContainer] = useState<X8dContainerDirectory | null>(null);
  const [omniFetching, setOmniFetching] = useState<{ active: boolean; lastMessage: string; error: string | null }>({
    active: false,
    lastMessage: 'Idle',
    error: null,
  });
  const [omniActionError, setOmniActionError] = useState<string | null>(null);
  const [omniInferPrompt, setOmniInferPrompt] = useState<string>('GR00T|env=bimanual_panda_gripper');
  const [omniInferResult, setOmniInferResult] = useState<OmniDiskInferResult | null>(null);
  const [isInferringDisk, setIsInferringDisk] = useState<boolean>(false);

  const fetchOmniStatus = useCallback(async () => {
    try {
      const resp = await fetch('/api/omni/status');
      if (!resp.ok) return;
      const data = (await resp.json()) as {
        manifest: OmniDatasetManifest | null;
        training: OmniTrainingMetricsSnapshot;
        fetching: { active: boolean; lastMessage: string; error: string | null };
        containerDirectory?: X8dContainerDirectory | null;
      };
      setOmniManifest(data.manifest);
      setOmniTraining(data.training);
      setOmniFetching(data.fetching);
      setOmniContainer(data.containerDirectory ?? null);
    } catch {
      // Transient fetch error ignored during poll
    }
  }, []);

  useEffect(() => {
    void fetchOmniStatus();
  }, [fetchOmniStatus]);

  useEffect(() => {
    const shouldPoll = omniTraining?.status === 'running' || omniFetching.active;
    if (!shouldPoll) return;
    const id = window.setInterval(() => {
      void fetchOmniStatus();
    }, 1500);
    return () => window.clearInterval(id);
  }, [omniTraining?.status, omniFetching.active, fetchOmniStatus]);

  const handleFetchOmniDatasets = async () => {
    setOmniActionError(null);
    setOmniFetching({ active: true, lastMessage: 'Fetching HuggingFace GR00T & Omni datasets...', error: null });
    try {
      const resp = await fetch('/api/omni/fetch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grootEnvs: 6, episodesPerEnv: 4, rowsPerDataset: 400 }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data?.error || 'Dataset fetch failed');
      }
      setOmniManifest(data.manifest);
      setOmniTraining(data.training);
      setOmniFetching(data.fetching);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to fetch HuggingFace datasets';
      setOmniActionError(msg);
      setOmniFetching({ active: false, lastMessage: 'Fetch failed', error: msg });
    }
  };

  const handleStartOmniTraining = async () => {
    setOmniActionError(null);
    try {
      const targetSteps = (omniTraining?.step ?? 0) + 240;
      const resp = await fetch('/api/omni/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ targetSteps, batchWindows: 24 }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.error || 'Could not start training');
      setOmniTraining(data.training);
    } catch (err: unknown) {
      setOmniActionError(err instanceof Error ? err.message : 'Failed to start training');
    }
  };

  const handleStopOmniTraining = async () => {
    setOmniActionError(null);
    try {
      const resp = await fetch('/api/omni/train/stop', { method: 'POST' });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.error || 'Could not stop training');
      setOmniTraining(data.training);
      await fetchOmniStatus();
    } catch (err: unknown) {
      setOmniActionError(err instanceof Error ? err.message : 'Failed to stop training');
    }
  };

  const handleQueryDiskX8d = async (e?: React.FormEvent, overridePrompt?: string) => {
    if (e) e.preventDefault();
    const promptToRun = overridePrompt ?? omniInferPrompt;
    setOmniActionError(null);
    setIsInferringDisk(true);
    try {
      const resp = await fetch('/api/omni/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptToRun, genBytes: 48 }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.error || 'Disk .x8D inference failed');
      setOmniInferResult(data as OmniDiskInferResult);
    } catch (err: unknown) {
      setOmniActionError(err instanceof Error ? err.message : 'Failed reading .x8D checkpoint');
    } finally {
      setIsInferringDisk(false);
    }
  };

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

  const handleRunNativeCompactTraining = () => {
    if (isTrainingNative) return;
    setIsTrainingNative(true);
    const params =
      nativeConfigId === 'k4-b4'
        ? { epochs: 32, kTerms: 4, factorBits: 4 as const }
        : nativeConfigId === 'k2-b4'
          ? { epochs: 32, kTerms: 2, factorBits: 4 as const }
          : { epochs: 32, kTerms: 2, factorBits: 2 as const };
    window.setTimeout(() => {
      try {
        const res = runNativeCompactTrainingExperiment(params);
        setNativeReport(res);
      } finally {
        setIsTrainingNative(false);
      }
    }, 15);
  };

  const snakeBodyKeys = useMemo(
    () => new Set(snakeState.snake.slice(1).map((p) => `${p.x},${p.y}`)),
    [snakeState.snake]
  );

  return (
    <section id="unified-arena" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          02. Byte Input Representation &amp; Train/Test Micro-Model (One Unified Mixture)
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Default 8-Bit Input → 0.000–0.255 Space · Train &amp; Test One Model Across Sudoku, Snake &amp; Q&amp;A
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Inputs from prompts or datasets map directly (<span className="font-mono text-slate-800">byte × 0.001</span>)
          into <span className="font-mono text-slate-800">0.000–0.255</span> instead of 50,257-entry GPT-2 BPE or
          discrete BLT IDs. Train the unified micro-weights below and run that same model on{' '}
          <strong className="font-semibold text-slate-900">System 2</strong> Sudoku,{' '}
          <strong className="font-semibold text-slate-900">System 1</strong> decision-by-decision Snake, and{' '}
          <strong className="font-semibold text-slate-900">Prompt Q&amp;A</strong> decoded back to UTF-8 text.
        </p>
      </div>

      {/* Input Encoding Comparison */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-slate-900">
            Input Representation: TinyStories Byte-BPE vs. Meta BLT (0..255) vs. ncm-mlx vs. x8D (0.000–0.255)
          </h3>
          <span className="text-xs font-mono text-slate-500">Default 8-bit Prompt / Dataset Input</span>
        </div>

        <div>
          <label htmlFor="tok-compare-input" className="block text-[11px] font-mono uppercase tracking-wider text-slate-500 mb-1.5">
            Live Input Text
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
              Needs JSON vocab + contraction regex. Splits digits, grids, and robot actions awkwardly.
            </p>
            <div className="mt-2 font-mono text-[11px] text-slate-700 break-all">
              Chunks: [{encodingCmp.bpeChunks.slice(0, 5).map((c) => JSON.stringify(c)).join(', ')}]
            </div>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3">
            <div className="font-mono text-[11px] font-semibold text-slate-700">2. Meta BLT</div>
            <div className="mt-1 font-mono text-sm font-semibold text-slate-900 tabular-nums">
              {encodingCmp.rawByteLength} bytes · IDs 0..255
            </div>
            <p className="mt-1 text-[11px] text-slate-600">
              Raw integer IDs <span className="font-mono">0..255</span> routed into a separate entropy patcher model.
            </p>
            <div className="mt-2 font-mono text-[11px] text-slate-700 break-all">
              BLT IDs: [{encodingCmp.bltIntIds.slice(0, 7).join(', ')}...]
            </div>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3">
            <div className="font-mono text-[11px] font-semibold text-slate-700">3. ncm-mlx Char + VQ</div>
            <div className="mt-1 font-mono text-sm font-semibold text-slate-900 tabular-nums">
              {encodingCmp.ncmCharVocabSize} alphabet · 2-Stage
            </div>
            <p className="mt-1 text-[11px] text-slate-600">
              Fixed padded char alphabet requiring a separate Stage-1 VQ codec before Stage-2 NCM.
            </p>
            <div className="mt-2 font-mono text-[11px] text-slate-700">
              Separate VQ tokenizer required
            </div>
          </div>

          <div className="rounded border border-blue-300 bg-blue-50/50 p-3">
            <div className="font-mono text-[11px] font-semibold text-blue-800">4. x8D Unified (0.000–0.255)</div>
            <div className="mt-1 font-mono text-sm font-semibold text-blue-900 tabular-nums">
              {encodingCmp.rawByteLength} bytes · 0.000..0.255
            </div>
            <p className="mt-1 text-[11px] text-slate-700">
              Every 8-bit input byte enters via <span className="font-mono">byte × 0.001</span> into one NCM-Diffusion loop without BPE or VQ.
            </p>
            <div className="mt-2 font-mono text-[11px] text-blue-800 break-all">
              Quanta: [{encodingCmp.x8dQuantaIds.slice(0, 6).join(', ')}...]
            </div>
          </div>
        </div>
      </div>

      {/* Train Mini Unified Model Strip */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm font-semibold text-slate-900">
              Unified 0.000–0.255 Micro-Model Training
            </span>
            <span
              className={`font-mono text-xs font-semibold ${
                weights.trainedEpochs > 0 ? 'text-emerald-700' : 'text-amber-700'
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

      {/* Three Live Test Modes Using THE ONE Model */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Sudoku Solver */}
        <div className="lg:col-span-4 min-w-0 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between pb-2.5 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-900">9×9 Sudoku (System 2 Loop)</h3>
            <span className="font-mono text-[11px] text-blue-700">0.049..0.057</span>
          </div>

          <div className="flex gap-2 min-w-0">
            <select
              aria-label="Sudoku Preset"
              value={selectedSudokuId}
              onChange={(e) => {
                setSelectedSudokuId(e.target.value);
                setSudokuResult(null);
              }}
              className="flex-1 min-w-0 rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900"
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
                {sudokuResult.solved ? 'Solved 0 Conflicts' : 'Unsolved'} · {sudokuResult.iterationsUsed} NCM+Diff Loops
              </div>
              <div className="font-mono text-[11px] text-slate-700 break-all">
                Row-0 Quanta: [{sudokuResult.outputQuantaSample.join(', ')}]
              </div>
            </div>
          ) : (
            <div className="text-[11px] font-mono text-slate-500">
              Click &quot;Solve&quot; or &quot;Train&quot; to solve in-loop over 0.049..0.057 digit bytes.
            </div>
          )}
        </div>

        {/* Decision-by-Decision Snake Game */}
        <div className="lg:col-span-4 min-w-0 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between pb-2.5 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-900">Snake (System 1 Expert · Iter 1)</h3>
            <span className="font-mono text-[11px] text-emerald-700">
              Score {snakeState.score} · Tick {snakeState.tick}
            </span>
          </div>

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
            <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-700 pb-1 border-b border-slate-200">
              <span data-testid="snake-food-coord">Food @({snakeState.food.x},{snakeState.food.y})</span>
              <span>Reach: {snakeState.lastReachableCells} cells</span>
              <span>Cycle-Breaks: {snakeState.loopBreaks}</span>
            </div>
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
                Dynamic PRNG food + anti-loop flood-fill active. Click &quot;Step +1 Decision&quot; or &quot;Auto&quot;.
              </div>
            )}
          </div>
        </div>

        {/* Prompt / Question Solving in 0.000..0.255 */}
        <div className="lg:col-span-4 min-w-0 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between pb-2.5 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-900">Prompt &amp; Question Solving</h3>
            <span className="font-mono text-[11px] text-slate-500">0.000–0.255 → UTF-8</span>
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
              Run Prompt in Unified Model
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
                  Inspect 0.000–0.255 I/O Quanta Trace
                </summary>
                <div className="mt-1.5 space-y-1 text-blue-700 break-all">
                  <div>In Quanta: [{qaResult.inputQuanta.slice(0, 8).join(', ')}]</div>
                  <div>Out Quanta: [{qaResult.answerQuanta.slice(0, 8).join(', ')}]</div>
                </div>
              </details>
            </div>
          )}
        </div>
      </div>

      {/* Decisive Experiment: Train Compact Native .x8D vs Dense from Random Init + Reload Only .x8D */}
      <div
        className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden"
        data-testid="native-compact-experiment"
      >
        <div className="px-5 py-4 bg-slate-50 border-b border-slate-200 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-blue-700 shrink-0" />
              <h3 className="text-sm font-semibold text-slate-900">
                From-Scratch Native Compact Training vs. Dense Reference (Zero Post-Hoc Compression)
              </h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Initializes BOTH models randomly (PPL ~256 over 256 byte states), trains directly on raw{' '}
              <span className="font-mono text-slate-800">0.000–0.255</span> bytes using Adam + Straight-Through Estimator
              (STE), exports ONLY packed binary <span className="font-mono text-slate-800">.x8D</span> factors{' '}
              (<span className="font-mono text-slate-800">Uint8Array</span>), discards training buffers, and reloads strictly
              from those bytes to evaluate held-out perplexity &amp; generation.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <select
              aria-label="Compact Native Kronecker Configuration"
              value={nativeConfigId}
              onChange={(e) => setNativeConfigId(e.target.value as 'k4-b4' | 'k2-b4' | 'k2-b2')}
              className="rounded border border-slate-300 bg-white px-2.5 py-2 text-xs font-mono text-slate-900"
            >
              <option value="k4-b4">Native K=4 · 4-bit STE</option>
              <option value="k2-b4">Native K=2 · 4-bit STE</option>
              <option value="k2-b2">Native K=2 · 2-bit STE</option>
            </select>
            <button
              type="button"
              data-testid="run-native-train-btn"
              onClick={handleRunNativeCompactTraining}
              disabled={isTrainingNative}
              className="inline-flex items-center justify-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-60 cursor-pointer"
            >
              <Zap className="w-3.5 h-3.5" />
              <span>{isTrainingNative ? 'Training From Step 0...' : 'Train & Reload .x8D Binary'}</span>
            </button>
          </div>
        </div>

        {nativeReport ? (
          <div className="p-5 space-y-4">
            <div className="overflow-x-auto border border-slate-200 rounded-md">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                    <th className="py-2.5 px-3">Model (Step 0 Init)</th>
                    <th className="py-2.5 px-3 text-right">Eff. Weights</th>
                    <th className="py-2.5 px-3 text-right">Trainable DOFs</th>
                    <th className="py-2.5 px-3 text-right">Serialized Bytes</th>
                    <th className="py-2.5 px-3 text-right">Measured BPW</th>
                    <th className="py-2.5 px-3 text-right">Train Loss</th>
                    <th className="py-2.5 px-3 text-right">Held-Out PPL</th>
                    <th className="py-2.5 px-3 text-right">Top-1 Acc</th>
                    <th className="py-2.5 px-3 text-right">Eval RAM</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 text-xs">
                  {[nativeReport.denseReference, nativeReport.compactNativeX8d].map((m) => {
                    const isCompact = m.modelId === 'compact-native-x8d';
                    return (
                      <tr key={m.modelId} className={isCompact ? 'bg-blue-50/50' : 'bg-white'}>
                        <td className="py-2.5 px-3">
                          <div className="font-semibold text-slate-900">{m.label}</div>
                          <div className="font-mono text-[11px] text-slate-500">{m.parameterization}</div>
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                          {m.effectiveWeights.toLocaleString()}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                          {m.trainableDofs.toLocaleString()}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-slate-900 whitespace-nowrap" data-testid={`native-bytes-${m.modelId}`}>
                          {m.serializedBytes.toLocaleString()} B
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-blue-800 font-semibold whitespace-nowrap">
                          {m.achievedBpw.toFixed(4)}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                          {m.initTrainLoss.toFixed(2)} → {m.finalTrainLoss.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-emerald-700 whitespace-nowrap" data-testid={`native-ppl-${m.modelId}`}>
                          {m.initHeldOutPpl.toFixed(0)} → {m.finalHeldOutPpl.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                          {m.heldOutTop1AccPct.toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                          {m.activeEvalBufferBytes.toLocaleString()} B
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="rounded border border-slate-200 bg-slate-50 p-3 space-y-1.5">
                <div className="font-mono text-[11px] font-semibold text-slate-700">
                  Dense Reference Continuation (Seed: <span className="text-slate-900">{nativeReport.seedPrompt}</span>)
                </div>
                <div className="font-mono text-xs text-slate-900 bg-white border border-slate-200 rounded px-2.5 py-1.5 break-all">
                  {nativeReport.denseReference.generatedContinuation}
                </div>
                <div className="font-mono text-[11px] text-slate-500 break-all">
                  Quanta: [{nativeReport.denseReference.generatedQuantaSample.join(', ')}]
                </div>
              </div>

              <div className="rounded border border-blue-200 bg-blue-50/40 p-3 space-y-1.5">
                <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
                  <span className="font-semibold text-blue-900">
                    Reloaded-Only .x8D Output ({nativeReport.compactNativeX8d.serializedBytes.toLocaleString()} B Buffer)
                  </span>
                  <span className="text-emerald-700">
                    Parity diff: {nativeReport.compactNativeX8d.reloadExactParityMaxDiff}
                  </span>
                </div>
                <div
                  className="font-mono text-xs text-slate-900 bg-white border border-blue-200 rounded px-2.5 py-1.5 break-all"
                  data-testid="native-x8d-generation"
                >
                  {nativeReport.compactNativeX8d.generatedContinuation}
                </div>
                <div className="font-mono text-[11px] text-blue-800 break-all">
                  Quanta: [{nativeReport.compactNativeX8d.generatedQuantaSample.join(', ')}]
                </div>
              </div>
            </div>

            <div className="rounded border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-xs text-slate-700 leading-relaxed">
              <strong className="font-semibold text-slate-900">Honest Measurement Note: </strong>
              {nativeReport.honestGapNote}
            </div>
          </div>
        ) : (
          <div className="px-5 py-4 text-xs font-mono text-slate-600">
            Click &quot;Train &amp; Reload .x8D Binary&quot; to train both Dense FP16 and Compact Native Kronecker (STE) from random initialization across {24_832} effective weights on disjoint train ({339} B) / held-out ({168} B) byte splits.
          </div>
        )}
      </div>

      {/* Live Full-Corpus Omni Training & Strict fs.readSync .x8D Disk Runtime */}
      <div
        className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden"
        data-testid="live-omni-training-console"
      >
        <div className="px-5 py-4 bg-slate-50 border-b border-slate-200 flex flex-col xl:flex-row xl:items-center xl:justify-between gap-4">
          <div className="space-y-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Database className="w-4 h-4 text-blue-700 shrink-0" />
              <h3 className="text-sm font-semibold text-slate-900">
                Live Full-Corpus Unified Omni Diffusion + NCM Training (GR00T + LeRobot + TinyStories + GSM8K + HumanEval + Sudoku)
              </h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Trains ONE{' '}
              <span className="font-mono text-slate-800">
                {(omniTraining?.effectiveWeights ?? 4_999_168).toLocaleString()}
              </span>{' '}
              (5.00M) nominal-weight unified Kronecker-MoE 256-state uniform-corruption diffusion model (
              <span className="font-mono text-slate-800">
                {(omniTraining?.trainableDofs ?? 24_320).toLocaleString()}
              </span>{' '}
              independent factor DOFs, 4 routed experts, System-1/System-2 NCM early-exit loop, zero <span className="font-mono">MASK=256</span>)
              directly on decontaminated HuggingFace shards on disk, writes a Safetensors-equivalent magic-free{' '}
              <span className="font-mono text-slate-800">.x8D v2</span> container (<span className="font-mono">x8d_arith</span> + 2/3-bit STE + CRC32), and executes strictly via{' '}
              <span className="font-mono text-slate-800">fs.openSync + fs.readSync</span> (no <span className="font-mono">mmap</span>).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              type="button"
              data-testid="omni-fetch-btn"
              onClick={handleFetchOmniDatasets}
              disabled={omniFetching.active || omniTraining?.status === 'running'}
              className="inline-flex items-center justify-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-100 disabled:opacity-50 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>{omniFetching.active ? 'Fetching HF Shards...' : 'Fetch / Expand HF Datasets'}</span>
            </button>

            {omniTraining?.status === 'running' ? (
              <button
                type="button"
                data-testid="omni-stop-btn"
                onClick={handleStopOmniTraining}
                className="inline-flex items-center justify-center gap-1.5 rounded-md bg-rose-700 px-3.5 py-2 text-xs font-semibold text-white hover:bg-rose-800 cursor-pointer"
              >
                <Square className="w-3.5 h-3.5" />
                <span>Stop &amp; Save .x8D</span>
              </button>
            ) : (
              <button
                type="button"
                data-testid="omni-start-btn"
                onClick={handleStartOmniTraining}
                disabled={omniFetching.active || !omniManifest}
                className="inline-flex items-center justify-center gap-1.5 rounded-md bg-blue-700 px-3.5 py-2 text-xs font-semibold text-white hover:bg-blue-800 disabled:opacity-50 cursor-pointer"
              >
                <Play className="w-3.5 h-3.5" />
                <span>Start / Resume Live Training</span>
              </button>
            )}
          </div>
        </div>

        {(omniActionError || omniTraining?.lastError || omniFetching.error) && (
          <div className="mx-5 mt-4 rounded border border-rose-200 bg-rose-50 px-3.5 py-2 text-xs font-mono text-rose-800">
            {omniActionError || omniTraining?.lastError || omniFetching.error}
          </div>
        )}

        <div className="p-5 space-y-5">
          {/* Telemetry Summary Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs">
            <div className="rounded border border-slate-200 bg-slate-50 p-3">
              <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">Trainer Status</div>
              <div
                className={`mt-1 font-mono text-sm font-bold uppercase ${
                  omniTraining?.status === 'running'
                    ? 'text-blue-700'
                    : omniTraining?.status === 'completed' || omniTraining?.status === 'paused'
                      ? 'text-emerald-700'
                      : 'text-slate-700'
                }`}
                data-testid="omni-status-label"
              >
                {omniTraining?.status ?? 'loading'}
              </div>
              <div className="mt-0.5 font-mono text-[11px] text-slate-600 tabular-nums" data-testid="omni-step-counter">
                Step {omniTraining?.step ?? 0} / {omniTraining?.targetSteps ?? 800}
              </div>
            </div>

            <div className="rounded border border-slate-200 bg-slate-50 p-3">
              <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">Train Loss</div>
              <div className="mt-1 font-mono text-sm font-bold text-slate-900 tabular-nums" data-testid="omni-loss-value">
                {omniTraining ? `${omniTraining.initialTrainLoss.toFixed(2)} → ${omniTraining.currentTrainLoss.toFixed(3)}` : '—'}
              </div>
              <div className="mt-0.5 font-mono text-[11px] text-slate-600 tabular-nums">
                {omniTraining?.epochsApprox.toFixed(3) ?? '0.000'} epochs
              </div>
            </div>

            <div className="rounded border border-slate-200 bg-slate-50 p-3">
              <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">CPU Throughput</div>
              <div className="mt-1 font-mono text-sm font-bold text-slate-900 tabular-nums">
                {omniTraining ? `${omniTraining.throughputBytesPerSec.toLocaleString()} B/s` : '—'}
              </div>
              <div className="mt-0.5 font-mono text-[11px] text-slate-600 tabular-nums">
                {(omniTraining?.bytesProcessed ?? 0).toLocaleString()} B trained
              </div>
            </div>

            <div className="rounded border border-slate-200 bg-slate-50 p-3">
              <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">Corpus &amp; #69 Dedup</div>
              <div className="mt-1 font-mono text-sm font-bold text-slate-900 tabular-nums" data-testid="omni-corpus-bytes">
                {omniManifest
                  ? `${((omniManifest.totalTrainBytes + omniManifest.totalHeldOutBytes) / 1024).toFixed(1)} KB`
                  : 'Not fetched'}
              </div>
              <div className="mt-0.5 font-mono text-[11px] text-slate-600 tabular-nums" data-testid="omni-dedup-status">
                {omniManifest?.dedupStats
                  ? `${omniManifest.dedupStats.residualShingleOverlapPct.toFixed(2)}% 24B overlap · -${omniManifest.dedupStats.exactDuplicatesRemoved} dup / -${omniManifest.dedupStats.heldOutContaminatedRemoved} leak`
                  : omniManifest
                    ? `${omniManifest.totalRecords.toLocaleString()} records`
                    : omniFetching.lastMessage}
              </div>
            </div>

            <div className="rounded border border-blue-200 bg-blue-50/50 p-3">
              <div className="font-mono text-[10px] uppercase tracking-wider text-blue-700">.x8D v2 File on Disk</div>
              <div className="mt-1 font-mono text-sm font-bold text-blue-900 tabular-nums" data-testid="omni-x8d-bytes">
                {omniTraining && omniTraining.checkpointBytesOnDisk > 0
                  ? `${omniTraining.checkpointBytesOnDisk.toLocaleString()} B`
                  : 'Unsaved'}
              </div>
              <div className="mt-0.5 font-mono text-[11px] text-blue-800 tabular-nums">
                {omniTraining && omniTraining.achievedBpw > 0
                  ? `${omniTraining.achievedBpw.toFixed(4)} bpw eff · ${omniTraining.achievedDofBpb.toFixed(2)} b/DOF`
                  : '5.00M eff · 24,320 DOFs'}
              </div>
            </div>

            <div className="rounded border border-slate-200 bg-slate-50 p-3">
              <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">4-Expert Router Load</div>
              <div className="mt-1 font-mono text-xs font-semibold text-slate-900 tabular-nums">
                {omniTraining
                  ? omniTraining.expertUsagePct.map((p, idx) => `E${idx}:${p}%`).join(' ')
                  : 'E0:25% E1:25% E2:25% E3:25%'}
              </div>
              <div className="mt-0.5 font-mono text-[11px] text-slate-600">
                {(omniTraining?.effectiveWeights ?? 4_999_168).toLocaleString()} eff ({(omniTraining?.trainableDofs ?? 24_320).toLocaleString()} DOFs)
              </div>
            </div>
          </div>

          {omniContainer && omniContainer.entries.length > 0 && (
            <details className="rounded-md border border-slate-200 bg-slate-50 px-4 py-2.5 text-xs" data-testid="omni-container-directory">
              <summary className="cursor-pointer font-mono text-[11px] font-semibold text-slate-800 flex flex-wrap items-center justify-between gap-2">
                <span>
                  .x8D v{omniContainer.version} Named-Tensor Directory ({omniContainer.tensorCount} spans · {omniContainer.fileSizeBytes.toLocaleString()} B · CRC32 verified · Zero X8DGGUF1 magic)
                </span>
                <span className="text-blue-700 font-normal">Inspect seek spans &amp; codecs</span>
              </summary>
              <div className="mt-3 overflow-x-auto border border-slate-200 rounded bg-white">
                <table className="w-full text-left border-collapse font-mono text-[11px]">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-[10px] uppercase tracking-wider text-slate-500">
                      <th className="py-1.5 px-2.5">Tensor</th>
                      <th className="py-1.5 px-2.5">Codec</th>
                      <th className="py-1.5 px-2.5 text-right">Shape (K)</th>
                      <th className="py-1.5 px-2.5 text-right">Seek Offset + Size</th>
                      <th className="py-1.5 px-2.5 text-right">CRC32</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {omniContainer.entries.map((entry) => (
                      <tr key={entry.name}>
                        <td className="py-1.5 px-2.5 font-semibold text-slate-900">{entry.name}</td>
                        <td className="py-1.5 px-2.5 text-blue-800">{formatCodecLabel(entry.codecId)}</td>
                        <td className="py-1.5 px-2.5 text-right text-slate-700 tabular-nums">
                          {entry.rows}×{entry.cols} (k={entry.kTerms})
                        </td>
                        <td className="py-1.5 px-2.5 text-right text-slate-700 tabular-nums">
                          @{entry.byteOffset} · {entry.byteLength.toLocaleString()} B
                        </td>
                        <td className="py-1.5 px-2.5 text-right text-emerald-700 tabular-nums">
                          0x{entry.crc32.toString(16).padStart(8, '0').toUpperCase()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}

          {/* Per-Dataset Table */}
          {omniManifest ? (
            <div className="overflow-x-auto border border-slate-200 rounded-md">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                    <th className="py-2.5 px-3">HuggingFace Source</th>
                    <th className="py-2.5 px-3">Modality / Split</th>
                    <th className="py-2.5 px-3 text-right">Records (Tr / Val)</th>
                    <th className="py-2.5 px-3 text-right">Raw Bytes on Disk</th>
                    <th className="py-2.5 px-3 text-right">Held-Out PPL (Init → Now)</th>
                    <th className="py-2.5 px-3 text-right">Top-1 Acc</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 text-xs">
                  {(
                    [
                      'groot_sim',
                      'lerobot_control',
                      'tinystories_prose',
                      'gsm8k_math',
                      'humaneval_code',
                      'sudoku_extreme',
                    ] as OmniDomainId[]
                  ).map((domId) => {
                    const dom = omniManifest.domains[domId];
                    const ppl = omniTraining?.perDomainHeldOutPpl?.[domId];
                    if (!dom) return null;
                    return (
                      <tr key={domId} className="bg-white hover:bg-slate-50/70">
                        <td className="py-2.5 px-3">
                          <div className="font-semibold text-slate-900">{dom.label}</div>
                          <div className="font-mono text-[11px] text-blue-700">{dom.sourceRepo}</div>
                        </td>
                        <td className="py-2.5 px-3 text-slate-700">
                          <div>{dom.modality}</div>
                          <div className="font-mono text-[10px] text-slate-500 truncate max-w-[260px]">
                            {dom.lastSamplePreview}
                          </div>
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                          {dom.recordsTrain.toLocaleString()} / {dom.recordsHeldOut.toLocaleString()}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                          {((dom.trainBytes + dom.heldOutBytes) / 1024).toFixed(1)} KB
                        </td>
                        <td
                          className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-emerald-700 whitespace-nowrap"
                          data-testid={`omni-ppl-${domId}`}
                        >
                          {ppl ? `${ppl.initPpl.toFixed(0)} → ${ppl.currentPpl.toFixed(2)}` : 'Pending eval'}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-slate-900 whitespace-nowrap">
                          {ppl ? `${ppl.top1AccPct.toFixed(1)}%` : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded border border-slate-200 bg-slate-50 p-4 text-xs font-mono text-slate-600">
              {omniFetching.active
                ? omniFetching.lastMessage
                : 'No omni datasets fetched on disk yet. Click "Fetch / Expand HF Datasets" above.'}
            </div>
          )}

          {/* Direct f.seek / f.read Query Panel Against omni_unified.x8d */}
          <div className="rounded-md border border-slate-200 bg-slate-50 p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-slate-800" />
                <h4 className="text-xs font-semibold text-slate-900">
                  Query Disk <span className="font-mono">.x8D</span> Only (<span className="font-mono">fs.openSync</span> +{' '}
                  <span className="font-mono">fs.readSync</span> Tail Directory Seek · Zero <span className="font-mono">mmap</span>)
                </h4>
              </div>
              <span className="font-mono text-[11px] text-slate-600">
                Reads shared spans + only the routed Expert span per step
              </span>
            </div>

            <form onSubmit={(e) => handleQueryDiskX8d(e)} className="flex flex-col sm:flex-row gap-2">
              <input
                type="text"
                aria-label="Prompt to infer strictly from omni_unified.x8d"
                value={omniInferPrompt}
                onChange={(e) => setOmniInferPrompt(e.target.value)}
                className="flex-1 min-w-0 rounded border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-900 focus:border-blue-600 focus:outline-none"
              />
              <button
                type="submit"
                data-testid="omni-infer-btn"
                disabled={isInferringDisk}
                className="inline-flex items-center justify-center gap-1.5 rounded bg-slate-900 px-4 py-2 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-50 cursor-pointer shrink-0"
              >
                <span>{isInferringDisk ? 'Seeking .x8D...' : 'Query Disk .x8D'}</span>
              </button>
            </form>

            <div className="flex flex-wrap gap-1.5">
              {[
                'GR00T|env=bimanual_panda_gripper',
                'LEROBOT|ds=pusht|ep=',
                'TEXT|Once upon a time',
                'MATH|Q:Janet has 3 apples',
                'CODE|HumanEval/0\ndef ',
                'SUDOKU|',
              ].map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => {
                    setOmniInferPrompt(preset);
                    void handleQueryDiskX8d(undefined, preset);
                  }}
                  className="rounded border border-slate-300 bg-white px-2 py-1 font-mono text-[11px] text-slate-700 hover:bg-slate-100 cursor-pointer"
                >
                  {preset}
                </button>
              ))}
            </div>

            {omniInferResult && (
              <div
                className="rounded border border-blue-200 bg-white p-3.5 space-y-2 text-xs"
                data-testid="omni-infer-result"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-slate-700 pb-2 border-b border-slate-200">
                  <span>
                    Disk File: <strong className="text-slate-900">{omniInferResult.diskFileSizeBytes.toLocaleString()} B</strong> ·{' '}
                    <strong className="text-blue-800">{omniInferResult.diskSeeksPerformed} f.readSync seeks</strong> (
                    {omniInferResult.diskBytesSeekRead.toLocaleString()} B read) ·{' '}
                    <strong className="text-emerald-700">{omniInferResult.crcVerifiedCount} CRC32 verified</strong>
                  </span>
                  <span>
                    S1 Iter-1 Exits: <strong className="text-emerald-700">{omniInferResult.system1EarlyExits}</strong> ·{' '}
                    S2 Refinements: <strong className="text-blue-700">{omniInferResult.system2Refinements}</strong> ·{' '}
                    Experts: <strong className="text-slate-900">[E{omniInferResult.activeExpertsUsed.join(', E')}]</strong> ·{' '}
                    {omniInferResult.latencyMs} ms
                  </span>
                </div>
                <div className="font-mono text-xs text-slate-900 break-all" data-testid="omni-infer-continuation">
                  <span className="text-slate-500">{omniInferResult.prompt}</span>
                  <span className="bg-blue-50 text-blue-900 px-1 rounded">{omniInferResult.continuation}</span>
                </div>
                <div className="font-mono text-[11px] text-slate-600 break-all">
                  0.000–0.255 Output Quanta: [{omniInferResult.outputQuantaSample.join(', ')}]
                </div>
              </div>
            )}
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-[11px] text-slate-700 leading-relaxed">
            <strong className="font-semibold text-slate-900">Honest Compute &amp; Scale Disclosure: </strong>
            This server runs on a 4-vCPU CPU-only sandbox (no GPU), training the real{' '}
            <span className="font-mono text-slate-800">
              {(omniTraining?.effectiveWeights ?? 4_999_168).toLocaleString()}
            </span>{' '}
            effective-weight (<span className="font-mono text-slate-800">{(omniTraining?.trainableDofs ?? 24_320).toLocaleString()}</span> independent DOF)
            256-state Uniform-Corruption Byte-Diffusion + NCM System-1/System-2 model on the{' '}
            <span className="font-mono text-slate-800">
              {omniManifest ? `${((omniManifest.totalTrainBytes + omniManifest.totalHeldOutBytes) / 1024 / 1024).toFixed(2)} MB` : '1.44 MB'}
            </span>{' '}
            decontaminated multi-modal HuggingFace disk corpus (including real binary GR00T{' '}
            <span className="font-mono">.parquet</span> trajectories). Scaling to multi-terabyte upstream archives requires
            distributed GPUs/TPUs; this workbench measures both{' '}
            <span className="font-mono text-slate-800">bpw vs. nominal effective weights</span> and{' '}
            <span className="font-mono text-slate-800">bits/DOF</span> directly from the checksummed{' '}
            <span className="font-mono">.x8D v2</span> binary file on disk with zero mocked metrics.
          </div>
        </div>
      </div>
    </section>
  );
};
