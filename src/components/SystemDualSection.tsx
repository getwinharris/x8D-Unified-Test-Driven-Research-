import React, { useMemo, useState } from 'react';
import { Globe, Loader2 } from 'lucide-react';
import {
  ByteBlockAnalysis,
  InternalRepresentationId,
  REPRESENTATION_OPTIONS,
  SYSTEM_COMPARISON_MATRIX,
  SystemTaskMode,
  TASK_PRESET_SAMPLES,
  UnifiedLoopConfig,
  simulateUnifiedLoop,
} from '../lib/unifiedX8dLoop';

const HF_DATASETS = [
  { id: 'lerobot/pusht', label: 'LeRobot PushT' },
  { id: 'nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim', label: 'GR00T Robotics' },
  { id: 'openai/gsm8k', label: 'GSM8K Math' },
  { id: 'roneneldan/TinyStories', label: 'TinyStories' },
];

export const SystemDualSection: React.FC = () => {
  const [config, setConfig] = useState<UnifiedLoopConfig>({
    taskMode: 'game-reflex',
    representationMode: 'adaptive-entropy-chunks',
    system1EntropyThreshold: TASK_PRESET_SAMPLES['game-reflex'].defaultThreshold,
    maxDiffusionIterations: 24,
    ncmCoarseLevels: 3,
  });

  const [payloadText, setPayloadText] = useState<string>(TASK_PRESET_SAMPLES['game-reflex'].sample);
  const [selectedBlockIdx, setSelectedBlockIdx] = useState<number>(0);

  // Live HF stream state
  const [hfDataset, setHfDataset] = useState<string>(HF_DATASETS[0].id);
  const [hfLoading, setHfLoading] = useState<boolean>(false);
  const [hfStatus, setHfStatus] = useState<string | null>(null);

  const sim = useMemo(() => simulateUnifiedLoop(payloadText, config), [payloadText, config]);

  const activeBlock: ByteBlockAnalysis | undefined =
    sim.blocks[selectedBlockIdx] ?? sim.blocks[0];

  const handleSelectTaskPreset = (mode: SystemTaskMode) => {
    const preset = TASK_PRESET_SAMPLES[mode];
    setConfig((prev) => ({
      ...prev,
      taskMode: mode,
      system1EntropyThreshold: preset.defaultThreshold,
    }));
    setPayloadText(preset.sample);
    setSelectedBlockIdx(0);
    setHfStatus(null);
  };

  const handleFetchHfSample = async () => {
    setHfLoading(true);
    setHfStatus(null);
    try {
      const resp = await fetch(`/api/hf/sample?dataset=${encodeURIComponent(hfDataset)}&row=0`);
      const data = await resp.json();
      if (!resp.ok) {
        setHfStatus(data?.error || `HTTP ${resp.status}`);
        return;
      }
      const text = typeof data.serializedText === 'string' ? data.serializedText : JSON.stringify(data.rawRecord);
      setPayloadText(text);
      setConfig((prev) => ({ ...prev, taskMode: 'omni-multimodal' }));
      setSelectedBlockIdx(0);
      setHfStatus(`Loaded ${hfDataset} (${text.length} bytes → 0.000..0.255)`);
    } catch (err: unknown) {
      setHfStatus(err instanceof Error ? err.message : 'Failed to fetch HF dataset row.');
    } finally {
      setHfLoading(false);
    }
  };

  return (
    <section id="system-dual" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          01. Prompt-to-Output NCM &amp; System 1 Decision Expert Inside Mixture
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Unified Mixture Architecture: Fast System 1 Expert + Deliberate System 2 NCM-Diffusion
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Next-Concept Modeling (NCM, <span className="font-mono text-slate-800">arXiv:2610.14162</span>) operates
          directly from Prompt to Output inside x8D&apos;s loop. A dedicated{' '}
          <strong className="font-semibold text-slate-900">System 1 decision expert lives inside the same mixture</strong>{' '}
          alongside AR/code/math/QA so robotics ticks, game reflexes, and deterministic PR #74 math commit in 1 pass,
          while hard reasoning refines iteratively.
        </p>
      </div>

      {/* Unified Loop Schematic Strip */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-slate-900">
            Prompt-to-Output NCM Loop Flow (0.000..0.255 Unified Byte Stream · Adaptive or Concept Spans)
          </h3>
          <span className="text-xs font-mono text-slate-500">Single Backbone · No Split Adapter Model</span>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="rounded border border-slate-200 bg-slate-50 p-3.5">
            <div className="font-mono text-[11px] font-semibold text-blue-700">1 · PROMPT INPUT (8-BIT)</div>
            <div className="mt-1 font-semibold text-slate-900">0..255 → 0.000..0.255</div>
            <p className="mt-1 text-slate-600 leading-relaxed">
              Prompt text, actions, and sensor structs convert via <span className="font-mono">byte × 0.001</span> into one shared byte space.
            </p>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3.5">
            <div className="font-mono text-[11px] font-semibold text-cyan-700">2 · NCM ITERATION 1</div>
            <div className="mt-1 font-semibold text-slate-900">Coarse Prompt→Output Blueprint</div>
            <p className="mt-1 text-slate-600 leading-relaxed">
              First NCM pass inside x8D drafts global structure across spans and evaluates entropy <span className="font-mono">H_norm</span>.
            </p>
          </div>

          <div className="rounded border border-emerald-200 bg-emerald-50/50 p-3.5">
            <div className="font-mono text-[11px] font-semibold text-emerald-800">3 · SYSTEM 1 DECISION EXPERT</div>
            <div className="mt-1 font-semibold text-slate-900">1-Step Exit Inside Mixture</div>
            <p className="mt-1 text-slate-700 leading-relaxed">
              Low-entropy control ticks, Snake decisions, robotics frames, and PR #74 Fraction math exit immediately on Iter 1.
            </p>
          </div>

          <div className="rounded border border-blue-200 bg-blue-50/50 p-3.5">
            <div className="font-mono text-[11px] font-semibold text-blue-800">4 · SYSTEM 2 &amp; OUTPUT</div>
            <div className="mt-1 font-semibold text-slate-900">Refine → UTF-8 / Action Decode</div>
            <p className="mt-1 text-slate-700 leading-relaxed">
              High-entropy logic spans refine across Iter 2..T and decode via <span className="font-mono">q / 0.001</span> back to language or actions.
            </p>
          </div>
        </div>
      </div>

      {/* System 1 vs System 2 Architectural Comparison Table */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-900">
            Architectural Comparison: Unified x8D vs. Pure AR, Pure Diffusion &amp; Split Dual-System
          </h3>
          <span className="text-xs font-mono text-slate-500">System 1 Inside Unified Mixture</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                <th className="py-3 px-4">Paradigm &amp; Token Domain</th>
                <th className="py-3 px-4">System 1 Role (Fast Decision)</th>
                <th className="py-3 px-4">System 2 Role (Deliberate Loop)</th>
                <th className="py-3 px-4">Control Latency</th>
                <th className="py-3 px-4">Reasoning &amp; Storage</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-xs">
              {SYSTEM_COMPARISON_MATRIX.map((row) => (
                <tr
                  key={row.id}
                  className={row.isOurHybrid ? 'bg-blue-50/50 align-top' : 'hover:bg-slate-50/70 align-top'}
                >
                  <td className="py-3.5 px-4 min-w-[230px]">
                    <div className="font-semibold text-slate-900">{row.paradigm}</div>
                    <div className="mt-1 text-[11px] text-slate-500">{row.examples}</div>
                    <div className="mt-1.5 font-mono text-[11px] text-blue-700">{row.tokenDomain}</div>
                  </td>
                  <td className="py-3.5 px-4 text-slate-700 leading-relaxed min-w-[200px]">{row.system1Role}</td>
                  <td className="py-3.5 px-4 text-slate-700 leading-relaxed min-w-[200px]">{row.system2Role}</td>
                  <td className="py-3.5 px-4 font-mono text-slate-800 leading-relaxed min-w-[170px]">
                    {row.gameAndControlLatency}
                  </td>
                  <td className="py-3.5 px-4 text-slate-700 leading-relaxed min-w-[210px]">
                    <div>{row.reasoningDepth}</div>
                    <div className="mt-1.5 font-mono text-[11px] text-slate-600">{row.modelAndMemoryCost}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Loop & Live HF Dataset Stream */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-5 min-w-0 bg-white border border-slate-200 rounded-lg p-5 space-y-5">
          <div>
            <div className="text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">
              Internal Span Grouping (8×8 Is Optional Baseline Only)
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
              {(Object.keys(REPRESENTATION_OPTIONS) as InternalRepresentationId[]).map((repId) => {
                const rep = REPRESENTATION_OPTIONS[repId];
                const active = (config.representationMode ?? 'adaptive-entropy-chunks') === repId;
                return (
                  <button
                    key={repId}
                    type="button"
                    data-testid={`rep-btn-${repId}`}
                    onClick={() => {
                      setConfig((prev) => ({ ...prev, representationMode: repId }));
                      setSelectedBlockIdx(0);
                    }}
                    className={`text-left rounded border p-2 text-xs transition-colors cursor-pointer ${
                      active
                        ? 'border-blue-600 bg-blue-50 text-blue-900'
                        : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-300'
                    }`}
                  >
                    <div className="font-semibold truncate">{rep.label}</div>
                    <div className="font-mono text-[10px] text-slate-500 mt-0.5">{rep.tag}</div>
                  </button>
                );
              })}
            </div>
            <p className="mt-1.5 text-[11px] text-slate-600 leading-relaxed">
              {REPRESENTATION_OPTIONS[config.representationMode ?? 'adaptive-entropy-chunks'].summary}
            </p>
          </div>

          <div className="pt-3 border-t border-slate-200 space-y-2.5">
            <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
              Load Preset or Live HuggingFace Dataset Row
            </div>
            <div className="flex flex-wrap gap-1.5">
              {(Object.keys(TASK_PRESET_SAMPLES) as SystemTaskMode[]).map((mode) => {
                const active = config.taskMode === mode;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => handleSelectTaskPreset(mode)}
                    className={`rounded border px-2 py-1 text-[11px] font-medium transition-colors cursor-pointer ${
                      active
                        ? 'border-blue-600 bg-blue-50 text-blue-800'
                        : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-300'
                    }`}
                  >
                    {TASK_PRESET_SAMPLES[mode].label}
                  </button>
                );
              })}
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center gap-2 pt-1 min-w-0">
              <select
                aria-label="Live HuggingFace dataset selector"
                data-testid="hf-dataset-select"
                value={hfDataset}
                onChange={(e) => setHfDataset(e.target.value)}
                className="w-full min-w-0 sm:flex-1 rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900 focus:border-blue-600 focus:outline-none"
              >
                {HF_DATASETS.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.label} ({ds.id})
                  </option>
                ))}
              </select>
              <button
                type="button"
                data-testid="fetch-hf-btn"
                onClick={handleFetchHfSample}
                disabled={hfLoading}
                className="inline-flex items-center gap-1.5 rounded bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-60 cursor-pointer shrink-0"
              >
                {hfLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Globe className="w-3.5 h-3.5" />}
                <span>{hfLoading ? 'Loading...' : 'Fetch Live HF'}</span>
              </button>
            </div>
            {hfStatus && (
              <div className="text-[11px] font-mono text-emerald-700" data-testid="hf-status">
                {hfStatus}
              </div>
            )}
          </div>

          <div>
            <label htmlFor="unified-loop-payload" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-1.5">
              Raw 8-Bit Payload (Mapped to 0.000..0.255)
            </label>
            <textarea
              id="unified-loop-payload"
              rows={4}
              value={payloadText}
              onChange={(e) => {
                setPayloadText(e.target.value);
                setSelectedBlockIdx(0);
              }}
              className="w-full rounded border border-slate-300 bg-white p-2.5 font-mono text-xs text-slate-900 focus:border-blue-600 focus:outline-none leading-relaxed"
            />
          </div>

          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <label htmlFor="s1-threshold-slider" className="font-mono uppercase tracking-wider text-slate-500">
                System 1 Exit Threshold (τ_S1)
              </label>
              <span className="font-mono font-semibold text-slate-900 tabular-nums">
                {config.system1EntropyThreshold.toFixed(2)}
              </span>
            </div>
            <input
              id="s1-threshold-slider"
              type="range"
              min={0.15}
              max={0.85}
              step={0.01}
              value={config.system1EntropyThreshold}
              onChange={(e) =>
                setConfig((prev) => ({ ...prev, system1EntropyThreshold: Number(e.target.value) }))
              }
              className="w-full accent-blue-600 cursor-pointer"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <label htmlFor="max-diff-iters" className="font-mono uppercase tracking-wider text-slate-500">
                  Max System 2 Iters
                </label>
                <span className="font-mono font-semibold text-slate-900 tabular-nums">
                  {config.maxDiffusionIterations}
                </span>
              </div>
              <input
                id="max-diff-iters"
                type="range"
                min={4}
                max={48}
                step={2}
                value={config.maxDiffusionIterations}
                onChange={(e) =>
                  setConfig((prev) => ({ ...prev, maxDiffusionIterations: Number(e.target.value) }))
                }
                className="w-full accent-blue-600 cursor-pointer"
              />
            </div>

            <div>
              <div className="text-xs font-mono uppercase tracking-wider text-slate-500 mb-1.5">
                NCM Coarse Levels
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {[2, 3, 4].map((lvl) => (
                  <button
                    key={lvl}
                    type="button"
                    onClick={() => setConfig((prev) => ({ ...prev, ncmCoarseLevels: lvl }))}
                    className={`rounded border py-1.5 font-mono text-xs cursor-pointer ${
                      config.ncmCoarseLevels === lvl
                        ? 'border-blue-600 bg-blue-50 text-blue-800 font-semibold'
                        : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    {lvl}L
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right Telemetry + Block Routing Grid */}
        <div className="lg:col-span-7 min-w-0 space-y-5">
          <div className="grid grid-cols-2 sm:grid-cols-4 border border-slate-200 bg-white rounded-lg divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">System 1 Fast Exit</div>
              <div className="mt-1 text-2xl font-mono font-semibold text-emerald-700 tabular-nums">
                {sim.system1Pct.toFixed(0)}%
              </div>
              <div className="mt-1 text-[11px] font-mono text-slate-600 tabular-nums">
                {sim.system1Blocks}/{sim.totalBlocks} spans @ 1 iter
              </div>
            </div>

            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">Avg Loop Iters</div>
              <div className="mt-1 text-2xl font-mono font-semibold text-slate-900 tabular-nums">
                {sim.avgIterationsPerBlock.toFixed(1)}×
              </div>
              <div className="mt-1 text-[11px] font-mono text-blue-700 tabular-nums">
                {sim.speedupVsFullDiffusion.toFixed(1)}× vs 48-step
              </div>
            </div>

            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">Est. Loop Latency</div>
              <div className="mt-1 text-2xl font-mono font-semibold text-slate-900 tabular-nums">
                {sim.estimatedLatencyMs.toFixed(1)} ms
              </div>
              <div className="mt-1 text-[11px] font-mono text-slate-600 tabular-nums">
                {config.ncmCoarseLevels}-level NCM carry
              </div>
            </div>

            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">x8d_arith Frame</div>
              <div className="mt-1 text-2xl font-mono font-semibold text-slate-900 tabular-nums">
                {sim.compressedBytesArith} B
              </div>
              <div className="mt-1 text-[11px] font-mono text-emerald-700 tabular-nums">
                {sim.losslessRoundtripVerified ? '100% Bit-Exact' : 'Mismatch'} ({sim.compressionRatio.toFixed(2)}×)
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
              <h3 className="text-sm font-semibold text-slate-900">
                Span Routing ({REPRESENTATION_OPTIONS[config.representationMode ?? 'adaptive-entropy-chunks'].label} · {sim.totalBytes}B → {sim.totalBlocks} spans)
              </h3>
              <div className="flex items-center gap-4 text-[11px] font-mono">
                <span className="flex items-center gap-1.5 text-emerald-800">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm bg-emerald-500" />
                  System 1 (Iter 1)
                </span>
                <span className="flex items-center gap-1.5 text-blue-800">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm bg-blue-600" />
                  System 2 (Iter 2..T)
                </span>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              {sim.blocks.map((blk) => {
                const isSelected = activeBlock?.blockIndex === blk.blockIndex;
                const isS1 = blk.routedSystem === 'system-1';
                return (
                  <button
                    key={blk.blockIndex}
                    type="button"
                    onClick={() => setSelectedBlockIdx(blk.blockIndex)}
                    className={`text-left rounded border p-2.5 transition-colors cursor-pointer ${
                      isSelected
                        ? 'border-slate-900 ring-1 ring-slate-900'
                        : isS1
                          ? 'border-emerald-200 bg-emerald-50/40 hover:border-emerald-400'
                          : 'border-blue-200 bg-blue-50/40 hover:border-blue-400'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[11px] font-mono">
                      <span className="font-semibold text-slate-900">BLK #{blk.blockIndex}</span>
                      <span className={isS1 ? 'text-emerald-800 font-semibold' : 'text-blue-800 font-semibold'}>
                        {isS1 ? 'S1·1x' : `S2·${blk.iterationsUsed}x`}
                      </span>
                    </div>
                    <div className="mt-1.5 text-[11px] font-mono text-slate-600 tabular-nums">
                      H={blk.normalizedEntropy.toFixed(2)} · {blk.byteCount}B
                    </div>
                    <div className="mt-1.5 h-1.5 w-full rounded bg-slate-200 overflow-hidden">
                      <div
                        className={isS1 ? 'h-full bg-emerald-500' : 'h-full bg-blue-600'}
                        style={{ width: `${Math.round(blk.normalizedEntropy * 100)}%` }}
                      />
                    </div>
                  </button>
                );
              })}
            </div>

            {activeBlock && (
              <div className="mt-4 rounded border border-slate-200 bg-slate-50 p-3.5 text-xs">
                <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-slate-700">
                  <span>
                    <strong className="text-slate-900">Span #{activeBlock.blockIndex}</strong> ·{' '}
                    {activeBlock.routedSystem === 'system-1'
                      ? 'System 1 Fast Expert (Committed on NCM Iter 1)'
                      : `System 2 Iterative Loop (${activeBlock.iterationsUsed} Diffusion Steps)`}
                  </span>
                  <span className="tabular-nums">
                    H_norm={activeBlock.normalizedEntropy.toFixed(3)} · Global Shannon={sim.shannonBitsPerByte.toFixed(2)} b/B
                  </span>
                </div>
                <div className="mt-2 font-mono text-[11px] text-slate-600 break-all">
                  Native 0.000..0.255 I/O Quanta: [{activeBlock.sampleQuanta.join(', ')}]
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
