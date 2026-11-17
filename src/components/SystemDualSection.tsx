import React, { useMemo, useState } from 'react';
import {
  ByteBlockAnalysis,
  SYSTEM_COMPARISON_MATRIX,
  SystemTaskMode,
  TASK_PRESET_SAMPLES,
  UnifiedLoopConfig,
  simulateUnifiedLoop,
} from '../lib/unifiedX8dLoop';

interface SystemDualSectionProps {
  externalPayload?: string | null;
  externalLabel?: string | null;
}

export const SystemDualSection: React.FC<SystemDualSectionProps> = ({
  externalPayload,
  externalLabel,
}) => {
  const [config, setConfig] = useState<UnifiedLoopConfig>({
    taskMode: 'game-reflex',
    system1EntropyThreshold: TASK_PRESET_SAMPLES['game-reflex'].defaultThreshold,
    maxDiffusionIterations: 24,
    ncmCoarseLevels: 3,
  });

  const [payloadText, setPayloadText] = useState<string>(TASK_PRESET_SAMPLES['game-reflex'].sample);
  const [selectedBlockIdx, setSelectedBlockIdx] = useState<number>(0);

  // Sync when user sends a live HF payload from the HF workbench
  const [lastSyncedExternal, setLastSyncedExternal] = useState<string | null>(null);
  if (externalPayload && externalPayload !== lastSyncedExternal) {
    setLastSyncedExternal(externalPayload);
    setPayloadText(externalPayload);
    setConfig((prev) => ({
      ...prev,
      taskMode: 'omni-multimodal',
    }));
    setSelectedBlockIdx(0);
  }

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
  };

  return (
    <section id="system-dual" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          01. Unified One-Model System 1 + System 2 Inside x8D Loop
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          NCM Inside x8D&apos;s Denoising Loop: Single 0.000–0.255 Backbone for Reflex &amp; Reasoning
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Instead of splitting System 2 (slow VLM) and System 1 (separate DiT action model) like GR00T N1, our
          architecture embeds Next-Concept Modeling (NCM) <strong className="font-semibold text-slate-900">directly inside x8D&apos;s iterative 8×8 loop</strong>.
          Every modality shares native <span className="font-mono text-slate-800">0.000–0.255</span> byte quanta:
          confident predictions, game ticks, and robot controls exit at Iteration 1 (System 1 Expert), while
          high-entropy reasoning blocks stay inside the NCM + Diffusion self-conditioning loop.
        </p>
      </div>

      {/* Unified Loop Schematic Strip */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-slate-900">
            Single-Backbone Loop Execution Flow (0.000..0.255 Unified Token Space)
          </h3>
          <span className="text-xs font-mono text-slate-500">
            1 Unified .x8D Image · Shared KDA/MLA + AttnRes Weights
          </span>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="rounded border border-slate-200 bg-slate-50 p-3.5">
            <div className="font-mono text-[11px] font-semibold text-blue-700">STAGE A · UNIFIED INPUT</div>
            <div className="mt-1 font-semibold text-slate-900">Raw Bytes → 0.000..0.255</div>
            <p className="mt-1 text-slate-600 leading-relaxed">
              Text, game actions, audio PCM, and GR00T kinematics map identically (<span className="font-mono">65 → 0.065</span>) into 8×8 DSpark blocks—zero adapter bridge.
            </p>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3.5">
            <div className="font-mono text-[11px] font-semibold text-cyan-700">ITER 1 · NCM COARSE PASS</div>
            <div className="mt-1 font-semibold text-slate-900">In-Loop Concept Blueprint</div>
            <p className="mt-1 text-slate-600 leading-relaxed">
              First pass inside x8D predicts hierarchical coarse structure across all 64-byte blocks and measures per-block normalized entropy <span className="font-mono">H_norm</span>.
            </p>
          </div>

          <div className="rounded border border-emerald-200 bg-emerald-50/50 p-3.5">
            <div className="font-mono text-[11px] font-semibold text-emerald-800">BRANCH 1 · SYSTEM 1 EXPERT</div>
            <div className="mt-1 font-semibold text-slate-900">1-Step Exit (~2–5 ms)</div>
            <p className="mt-1 text-slate-700 leading-relaxed">
              When <span className="font-mono">H_norm ≤ τ_S1</span> (game frames, quick reflex predictions, control ticks, or PR #74 Fraction math), block commits on Iter 1.
            </p>
          </div>

          <div className="rounded border border-blue-200 bg-blue-50/50 p-3.5">
            <div className="font-mono text-[11px] font-semibold text-blue-800">BRANCH 2 · SYSTEM 2 LOOP</div>
            <div className="mt-1 font-semibold text-slate-900">NCM + Diffusion Iter 2..T</div>
            <p className="mt-1 text-slate-700 leading-relaxed">
              High-entropy planning, math, or grasp-contact blocks loop inside x8D using self-conditioning carry + Uni-LaDiR latent state until stable.
            </p>
          </div>
        </div>
      </div>

      {/* System 1 vs System 2 Architectural Comparison Table */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-900">
            System 1 vs. System 2 Models Comparison: Why One Unified x8D Loop Beats Split Dual-System Pipelines
          </h3>
          <span className="text-xs font-mono text-slate-500">4 Architectural Paradigms</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                <th className="py-3 px-4">Paradigm &amp; Token Domain</th>
                <th className="py-3 px-4">System 1 Role (Fast Reflex)</th>
                <th className="py-3 px-4">System 2 Role (Deliberate Loop)</th>
                <th className="py-3 px-4">Game / Control Latency</th>
                <th className="py-3 px-4">Reasoning &amp; Footprint</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-xs">
              {SYSTEM_COMPARISON_MATRIX.map((row) => (
                <tr
                  key={row.id}
                  className={row.isOurHybrid ? 'bg-blue-50/50 align-top' : 'hover:bg-slate-50/70 align-top'}
                >
                  <td className="py-3.5 px-4 min-w-[230px]">
                    <div className="font-semibold text-slate-900">
                      {row.paradigm}
                    </div>
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

      {/* Interactive Loop & Iteration Workbench */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5 space-y-5">
          <div>
            <div className="text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">
              Task Regime Preset
            </div>
            <div className="flex flex-wrap gap-2">
              {(Object.keys(TASK_PRESET_SAMPLES) as SystemTaskMode[]).map((mode) => {
                const active = config.taskMode === mode;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => handleSelectTaskPreset(mode)}
                    className={`rounded border px-2.5 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
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
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label htmlFor="unified-loop-payload" className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Multimodal Stream Payload (Any Language / Action / Code)
              </label>
              {lastSyncedExternal && externalLabel && payloadText === lastSyncedExternal && (
                <span className="text-[11px] font-mono text-emerald-700">Synced: {externalLabel}</span>
              )}
            </div>
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
                System 1 Exit Entropy Threshold (τ_S1)
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
            <div className="flex justify-between text-[11px] font-mono text-slate-500 mt-0.5">
              <span>0.15 (Strict System 2)</span>
              <span>0.85 (Aggressive System 1)</span>
            </div>
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

        {/* Right Telemetry + 8x8 Block Grid */}
        <div className="lg:col-span-7 space-y-5">
          <div className="grid grid-cols-2 sm:grid-cols-4 border border-slate-200 bg-white rounded-lg divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">System 1 Fast Exit</div>
              <div className="mt-1 text-2xl font-mono font-semibold text-emerald-700 tabular-nums">
                {sim.system1Pct.toFixed(0)}%
              </div>
              <div className="mt-1 text-[11px] font-mono text-slate-600 tabular-nums">
                {sim.system1Blocks}/{sim.totalBlocks} blocks @ 1 iter
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
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">.x8D Codec 0x01</div>
              <div className="mt-1 text-2xl font-mono font-semibold text-slate-900 tabular-nums">
                {sim.compressedBytesCodec01} B
              </div>
              <div className="mt-1 text-[11px] font-mono text-emerald-700 tabular-nums">
                {sim.losslessRoundtripVerified ? '● 100% Bit-Exact' : 'Mismatch'} ({sim.compressionRatio.toFixed(2)}×)
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
              <h3 className="text-sm font-semibold text-slate-900">
                Live DSpark 8×8 Block Routing Map ({sim.totalBytes} bytes → {sim.totalBlocks} block{sim.totalBlocks > 1 ? 's' : ''})
              </h3>
              <div className="flex items-center gap-4 text-[11px] font-mono">
                <span className="flex items-center gap-1.5 text-emerald-800">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm bg-emerald-500" />
                  System 1 (Iter 1 Exit)
                </span>
                <span className="flex items-center gap-1.5 text-blue-800">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm bg-blue-600" />
                  System 2 (NCM+Diff Loop)
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
                    <strong className="text-slate-900">Block #{activeBlock.blockIndex} Inspector</strong> ·{' '}
                    {activeBlock.routedSystem === 'system-1'
                      ? 'System 1 Fast Expert (Committed on NCM Iter 1)'
                      : `System 2 Iterative Loop (${activeBlock.iterationsUsed} Diffusion Steps)`}
                  </span>
                  <span className="tabular-nums">
                    H_norm={activeBlock.normalizedEntropy.toFixed(3)} · Global Shannon={sim.shannonBitsPerByte.toFixed(2)} b/B
                  </span>
                </div>
                <div className="mt-2 font-mono text-[11px] text-slate-600 break-all">
                  Native Unified Tokens (0.000..0.255): [{activeBlock.sampleQuanta.join(', ')}]
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
