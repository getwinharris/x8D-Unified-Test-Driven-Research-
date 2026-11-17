import React, { useMemo, useState } from 'react';
import { ArrowUpRight, Database, Loader2, Play } from 'lucide-react';
import { simulateUnifiedLoop, toX8dQuantaString } from '../lib/unifiedX8dLoop';
import { formatBytes } from '../lib/architectureMath';

interface HfDatasetSpec {
  id: string;
  label: string;
  modality: string;
  recommendedSystemMode: string;
}

const HF_DATASETS: HfDatasetSpec[] = [
  {
    id: 'lerobot/pusht',
    label: 'lerobot/pusht (Real-Time Game/Control)',
    modality: 'Action + State Telemetry',
    recommendedSystemMode: 'System 1 Fast Expert',
  },
  {
    id: 'lerobot/aloha_sim_insertion_human',
    label: 'lerobot/aloha_sim_insertion_human',
    modality: 'Bimanual Robotics Trajectory',
    recommendedSystemMode: 'Hybrid S1 + S2 Contact Loop',
  },
  {
    id: 'nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim',
    label: 'nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim',
    modality: 'GR00T Multi-Task Sim Index',
    recommendedSystemMode: 'GR00T Multimodal Pretraining',
  },
  {
    id: 'openai/gsm8k',
    label: 'openai/gsm8k (Multi-Step Math)',
    modality: 'Language + Symbolic Reasoning',
    recommendedSystemMode: 'System 2 NCM+Diffusion Loop',
  },
  {
    id: 'roneneldan/TinyStories',
    label: 'roneneldan/TinyStories (Coarse-to-Fine Text)',
    modality: 'Narrative Language',
    recommendedSystemMode: 'NCM Outline + S1 Fill',
  },
  {
    id: 'Salesforce/wikitext',
    label: 'Salesforce/wikitext (wikitext-2-raw-v1)',
    modality: 'Dense Encyclopedic Text',
    recommendedSystemMode: 'Mixed S1/S2 Byte Stream',
  },
];

interface HfSampleResponse {
  dataset: string;
  rowIndex: number;
  totalCachedRows?: number;
  sourceEndpoint: string;
  rawRecord: unknown;
  serializedText: string;
}

interface HfTestTrainSectionProps {
  onSendPayloadToLoop: (payload: string, sourceLabel: string) => void;
}

export const HfTestTrainSection: React.FC<HfTestTrainSectionProps> = ({
  onSendPayloadToLoop,
}) => {
  const [selectedDataset, setSelectedDataset] = useState<string>('lerobot/pusht');
  const [rowIdx, setRowIdx] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [sample, setSample] = useState<HfSampleResponse | null>(null);

  // Unified 50M Training Harness Parameters
  const [batchSize, setBatchSize] = useState<number>(64);
  const [seqBytes, setSeqBytes] = useState<number>(4096);
  const [system1CurriculumPct, setSystem1CurriculumPct] = useState<number>(45);
  const [optimizerMode, setOptimizerMode] = useState<'muon-adamw' | 'adamw'>('muon-adamw');
  const [gpuCount, setGpuCount] = useState<number>(4);
  const [checkpointMode, setCheckpointMode] = useState<'x8d-unified' | 'bf16'>('x8d-unified');
  const [epochStepsRun, setEpochStepsRun] = useState<number>(0);

  const handleFetchSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(
        `/api/hf/sample?dataset=${encodeURIComponent(selectedDataset)}&row=${rowIdx}`
      );
      const data = await resp.json();
      if (!resp.ok) {
        setError(data?.error || `HTTP ${resp.status} fetching HuggingFace dataset.`);
        return;
      }
      setSample(data as HfSampleResponse);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Network error reaching /api/hf/sample.');
    } finally {
      setLoading(false);
    }
  };

  // Analyze live HF payload directly in unified 0.000..0.255 space
  const liveTestMetrics = useMemo(() => {
    if (!sample) return null;
    const text = sample.serializedText;
    const threshold =
      sample.dataset === 'openai/gsm8k'
        ? 0.42
        : sample.dataset.startsWith('lerobot/') || sample.dataset.includes('TinyStories')
          ? 0.74
          : 0.68;
    const loopRes = simulateUnifiedLoop(text, {
      taskMode: 'omni-multimodal',
      system1EntropyThreshold: threshold,
      maxDiffusionIterations: 24,
      ncmCoarseLevels: 3,
    });
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    const previewTokens: Array<{ char: string; byte: number; quanta: string }> = [];
    for (let i = 0; i < Math.min(18, bytes.length); i++) {
      const b = bytes[i];
      const ch = b >= 32 && b <= 126 ? String.fromCharCode(b) : `0x${b.toString(16).padStart(2, '0')}`;
      previewTokens.push({ char: ch, byte: b, quanta: toX8dQuantaString(b) });
    }
    return { loopRes, previewTokens };
  }, [sample]);

  // Unified 50M FSDP/ZeRO-2 Training Resource Estimator
  const trainStats = useMemo(() => {
    const params = 51_251_200; // 50M Unified NCM-Diffusion backbone
    // Per-GPU sharded state (weights BF16 + grads BF16 + Muon/AdamW FP32 moments)
    const bytesPerParamSharded =
      optimizerMode === 'muon-adamw'
        ? (2 + 2 + 8) / gpuCount // Muon skips v_t for 2D matrix params
        : (2 + 2 + 12) / gpuCount;
    const modelAndOptVram = params * bytesPerParamSharded;

    // Activations with 14L, d=512, 3:1 KDA:MLA + gradient checkpointing
    const actBytesPerBatch = batchSize * seqBytes * 512 * 14 * 0.18; // ~18% retained via AttnRes checkpoints
    const perGpuVramBytes = modelAndOptVram + actBytesPerBatch / gpuCount;

    // Unified tokens (1 byte = 1 native 0.000..0.255 token)
    const tokensPerStep = batchSize * seqBytes;
    // NCM coarse skip saves ~35% compute in S1 curriculum spans
    const effectiveThroughputMultiplier = 1 + (system1CurriculumPct / 100) * 0.55;
    const estStepMs = Math.max(
      18,
      ((tokensPerStep / (gpuCount * 145_000)) * 1000) / effectiveThroughputMultiplier
    );
    const tokensPerSec = Math.round((tokensPerStep / estStepMs) * 1000);

    // Checkpoint disk size (0.008 bpw target: 50M -> ~50.05 KB)
    const checkpointDiskBytes =
      checkpointMode === 'x8d-unified'
        ? Math.round((params * 0.008) / 8)
        : params * 2; // BF16

    // Deterministic synthetic loss convergence trajectory anchored to live HF sample entropy if loaded
    const baseEntropy = liveTestMetrics ? liveTestMetrics.loopRes.shannonBitsPerByte : 4.2;
    const startLoss = Number((baseEntropy * 0.693 + 0.45).toFixed(3));
    const currentStep = epochStepsRun * 50;
    const currentLoss = Number(
      (0.82 + (startLoss - 0.82) * Math.exp(-currentStep / 380)).toFixed(3)
    );

    return {
      perGpuVramBytes,
      tokensPerStep,
      estStepMs,
      tokensPerSec,
      checkpointDiskBytes,
      startLoss,
      currentStep,
      currentLoss,
    };
  }, [
    batchSize,
    seqBytes,
    system1CurriculumPct,
    optimizerMode,
    gpuCount,
    checkpointMode,
    epochStepsRun,
    liveTestMetrics,
  ]);

  return (
    <section id="hf-workbench" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          02. Live HuggingFace Multimodal Testing &amp; 50M Training Harness
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Stream Real HuggingFace Rows into 0.000–0.255 Tokens &amp; Configure Unified Training
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Fetch real samples directly from HuggingFace (LeRobot robotics, GR00T simulation tasks, GSM8K math,
          TinyStories, WikiText), inspect their lossless <span className="font-mono text-slate-800">0.000–0.255</span> tokenization
          and <span className="font-mono text-slate-800">.x8D Codec 0x01</span> packing, and size your distributed 50M
          System 1 + System 2 training run.
        </p>
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Live HuggingFace Dataset Tester */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
              <Database className="w-4 h-4 text-blue-600" />
              <span>Live HuggingFace Dataset Probe</span>
            </div>
            <span className="text-xs font-mono text-slate-500">GET /api/hf/sample</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-end">
            <div className="sm:col-span-7">
              <label htmlFor="hf-dataset-select" className="block text-[11px] font-mono uppercase tracking-wider text-slate-500 mb-1">
                Dataset Repository
              </label>
              <select
                id="hf-dataset-select"
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full rounded border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-900 focus:border-blue-600 focus:outline-none"
              >
                {HF_DATASETS.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="sm:col-span-2">
              <label htmlFor="hf-row-input" className="block text-[11px] font-mono uppercase tracking-wider text-slate-500 mb-1">
                Row / Task #
              </label>
              <input
                id="hf-row-input"
                type="number"
                min={0}
                max={99}
                value={rowIdx}
                onChange={(e) => setRowIdx(Math.max(0, Math.min(99, Number(e.target.value) || 0)))}
                className="w-full rounded border border-slate-300 bg-white px-2.5 py-2 font-mono text-xs text-slate-900 tabular-nums focus:border-blue-600 focus:outline-none"
              />
            </div>

            <div className="sm:col-span-3">
              <button
                type="button"
                onClick={handleFetchSample}
                disabled={loading}
                className="w-full flex items-center justify-center gap-1.5 rounded bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-60 transition-colors cursor-pointer"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Fetching...</span>
                  </>
                ) : (
                  <span>Fetch Live HF Sample</span>
                )}
              </button>
            </div>
          </div>

          {error && (
            <div role="alert" className="rounded border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800">
              <strong className="font-mono">HF FETCH ERROR: </strong>
              {error}
            </div>
          )}

          {!sample && !loading && !error && (
            <div className="rounded border border-dashed border-slate-300 bg-slate-50 p-6 text-center">
              <p className="text-xs font-medium text-slate-800">
                Click &quot;Fetch Live HF Sample&quot; to pull an unmocked record from HuggingFace.
              </p>
              <p className="mt-1 text-[11px] font-mono text-slate-500">
                Supports LeRobot PushT, ALOHA, GR00T-X-Embodiment-Sim task trees, GSM8K, TinyStories &amp; WikiText.
              </p>
            </div>
          )}

          {sample && liveTestMetrics && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
                  <div className="text-[10px] font-mono uppercase text-slate-500">Raw Record Bytes</div>
                  <div className="mt-0.5 text-sm font-mono font-semibold text-slate-900 tabular-nums">
                    {liveTestMetrics.loopRes.totalBytes} B
                  </div>
                </div>
                <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
                  <div className="text-[10px] font-mono uppercase text-slate-500">.x8D Codec 0x01</div>
                  <div className="mt-0.5 text-sm font-mono font-semibold text-emerald-700 tabular-nums">
                    {liveTestMetrics.loopRes.compressedBytesCodec01} B (
                    {liveTestMetrics.loopRes.compressionRatio.toFixed(2)}×)
                  </div>
                </div>
                <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
                  <div className="text-[10px] font-mono uppercase text-slate-500">System 1 Exit</div>
                  <div className="mt-0.5 text-sm font-mono font-semibold text-blue-700 tabular-nums">
                    {liveTestMetrics.loopRes.system1Pct.toFixed(0)}% ({liveTestMetrics.loopRes.avgIterationsPerBlock.toFixed(1)}× avg)
                  </div>
                </div>
                <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
                  <div className="text-[10px] font-mono uppercase text-slate-500">Bit-Exact Check</div>
                  <div className="mt-0.5 text-sm font-mono font-semibold text-emerald-700">
                    {liveTestMetrics.loopRes.losslessRoundtripVerified ? '100% Exact' : 'Failed'}
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 mb-1">
                  <span>Native Unified Tokens (First 18 Bytes → 0.000..0.255)</span>
                  <button
                    type="button"
                    onClick={() => {
                      onSendPayloadToLoop(sample.serializedText, `${sample.dataset} #${sample.rowIndex}`);
                      document.getElementById('system-dual')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }}
                    className="inline-flex items-center gap-1 font-semibold text-blue-700 hover:underline cursor-pointer"
                  >
                    <span>Inspect in System 1/2 Loop</span>
                    <ArrowUpRight className="w-3 h-3" />
                  </button>
                </div>
                <div className="flex flex-wrap gap-1.5 rounded border border-slate-200 bg-slate-50 p-2.5 font-mono text-[11px]">
                  {liveTestMetrics.previewTokens.map((tok, i) => (
                    <span
                      key={i}
                      className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-slate-800 tabular-nums"
                    >
                      <span className="text-slate-400">{tok.char}→</span>
                      <span className="font-semibold text-blue-700">{tok.quanta}</span>
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 mb-1">
                  <span>Live JSON Payload ({sample.dataset} · row #{sample.rowIndex})</span>
                  <span className="truncate max-w-[260px]">{sample.sourceEndpoint}</span>
                </div>
                <pre className="max-h-44 overflow-auto rounded border border-slate-200 bg-slate-900 p-3 font-mono text-[11px] text-slate-100 leading-relaxed">
                  {JSON.stringify(sample.rawRecord, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>

        {/* Right: Unified 50M Training Harness */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-900">50M Unified Training Harness</h3>
            <span className="text-xs font-mono text-slate-500">14L · d=512 · 0.000..0.255</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label htmlFor="train-batch" className="block text-[11px] font-mono uppercase text-slate-500 mb-1">
                Global Batch Size
              </label>
              <select
                id="train-batch"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full rounded border border-slate-300 bg-white px-2.5 py-1.5 font-mono text-xs text-slate-900"
              >
                {[16, 32, 64, 128, 256].map((b) => (
                  <option key={b} value={b}>
                    {b} seqs/step
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="train-seqlen" className="block text-[11px] font-mono uppercase text-slate-500 mb-1">
                Byte Canvas Length
              </label>
              <select
                id="train-seqlen"
                value={seqBytes}
                onChange={(e) => setSeqBytes(Number(e.target.value))}
                className="w-full rounded border border-slate-300 bg-white px-2.5 py-1.5 font-mono text-xs text-slate-900"
              >
                {[1024, 2048, 4096, 8192, 16384].map((s) => (
                  <option key={s} value={s}>
                    {s.toLocaleString()} bytes
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="train-gpus" className="block text-[11px] font-mono uppercase text-slate-500 mb-1">
                FSDP Worker GPUs
              </label>
              <select
                id="train-gpus"
                value={gpuCount}
                onChange={(e) => setGpuCount(Number(e.target.value))}
                className="w-full rounded border border-slate-300 bg-white px-2.5 py-1.5 font-mono text-xs text-slate-900"
              >
                {[1, 2, 4, 8].map((g) => (
                  <option key={g} value={g}>
                    {g}× GPU (ZeRO-2)
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="train-opt" className="block text-[11px] font-mono uppercase text-slate-500 mb-1">
                Optimizer Regime
              </label>
              <select
                id="train-opt"
                value={optimizerMode}
                onChange={(e) => setOptimizerMode(e.target.value as 'muon-adamw' | 'adamw')}
                className="w-full rounded border border-slate-300 bg-white px-2.5 py-1.5 font-mono text-xs text-slate-900"
              >
                <option value="muon-adamw">Muon + AdamW</option>
                <option value="adamw">AdamW FP32</option>
              </select>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <label htmlFor="s1-curriculum-slider" className="font-mono uppercase text-slate-500">
                System 1 Reflex / Game Mix Ratio
              </label>
              <span className="font-mono font-semibold text-slate-900 tabular-nums">
                {system1CurriculumPct}% S1 · {100 - system1CurriculumPct}% S2
              </span>
            </div>
            <input
              id="s1-curriculum-slider"
              type="range"
              min={10}
              max={75}
              step={5}
              value={system1CurriculumPct}
              onChange={(e) => setSystem1CurriculumPct(Number(e.target.value))}
              className="w-full accent-blue-600 cursor-pointer"
            />
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="font-mono uppercase text-slate-500 text-[11px]">Checkpoint Target:</span>
            <button
              type="button"
              onClick={() => setCheckpointMode('x8d-unified')}
              className={`rounded border px-2.5 py-1 font-mono text-[11px] cursor-pointer ${
                checkpointMode === 'x8d-unified'
                  ? 'border-blue-600 bg-blue-50 text-blue-800 font-semibold'
                  : 'border-slate-200 bg-white text-slate-700'
              }`}
            >
              .x8D Unified (0.008 bpw)
            </button>
            <button
              type="button"
              onClick={() => setCheckpointMode('bf16')}
              className={`rounded border px-2.5 py-1 font-mono text-[11px] cursor-pointer ${
                checkpointMode === 'bf16'
                  ? 'border-blue-600 bg-blue-50 text-blue-800 font-semibold'
                  : 'border-slate-200 bg-white text-slate-700'
              }`}
            >
              BF16 (16 bpw)
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2.5 border-t border-slate-200 pt-3">
            <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-500">VRAM / GPU (FSDP)</div>
              <div className="mt-0.5 text-base font-mono font-semibold text-slate-900 tabular-nums">
                {formatBytes(trainStats.perGpuVramBytes)}
              </div>
            </div>
            <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-500">Throughput (0..0.255)</div>
              <div className="mt-0.5 text-base font-mono font-semibold text-slate-900 tabular-nums">
                {trainStats.tokensPerSec.toLocaleString()} tok/s
              </div>
            </div>
            <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-500">Checkpoint Disk</div>
              <div className="mt-0.5 text-base font-mono font-semibold text-emerald-700 tabular-nums">
                {formatBytes(trainStats.checkpointDiskBytes)}
              </div>
            </div>
            <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
              <div className="text-[10px] font-mono uppercase text-slate-500">Estimated Step Loss</div>
              <div className="mt-0.5 text-base font-mono font-semibold text-blue-700 tabular-nums">
                {trainStats.currentLoss.toFixed(3)}{' '}
                <span className="text-[11px] font-normal text-slate-500">(@{trainStats.currentStep})</span>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={() => setEpochStepsRun((n) => n + 1)}
            className="w-full flex items-center justify-center gap-2 rounded border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-100 transition-colors cursor-pointer"
          >
            <Play className="w-3.5 h-3.5 text-blue-600" />
            <span>Step Simulated 50M S1+S2 Curriculum (+50 steps · Joint Entropy CE + NCM Loss)</span>
          </button>
        </div>
      </div>
    </section>
  );
};
