import React, { useMemo } from 'react';
import {
  ExecutionModeId,
  FootprintSimulationInput,
  MODEL_SCALE_PRESETS,
  QUANTIZATION_REGIMES,
  QuantizationRegimeId,
  ScalePresetId,
  formatBytes,
  formatParams,
  simulateFootprint,
} from '../lib/architectureMath';

interface FootprintLabSectionProps {
  config: FootprintSimulationInput;
  onChange: (next: FootprintSimulationInput) => void;
}

const QUANT_IDS: QuantizationRegimeId[] = [
  'fp16',
  'int4',
  'u8-quanta-0001',
  'block-mean-0001',
  'x8d-subbyte-0008',
];

const SCALE_IDS: ScalePresetId[] = ['ncm-50m', 'ncm-150m', 'x8d-1b', 'x8d-1t'];

export const FootprintLabSection: React.FC<FootprintLabSectionProps> = ({ config, onChange }) => {
  const result = useMemo(() => simulateFootprint(config), [config]);

  const comparisonRows = useMemo(() => {
    return QUANT_IDS.map((qId) => {
      const sim = simulateFootprint({ ...config, quantId: qId });
      const spec = QUANTIZATION_REGIMES[qId];
      return { qId, spec, sim };
    });
  }, [config]);

  const maxDiskBytes = useMemo(
    () => Math.max(...comparisonRows.map((r) => r.sim.diskBytes), 1),
    [comparisonRows]
  );

  const selectedQuant = QUANTIZATION_REGIMES[config.quantId];

  return (
    <section id="footprint-lab" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          07. Memory, Compute &amp; Raw .x8D Direct-Read Simulator
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Quantization Footprint, Raw .x8D Direct Read (No mmap) &amp; Compute Lab
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Evaluate how reading directly from a &gt;93%–99.9% compressed raw <span className="font-mono text-slate-800">.x8D</span>{' '}
          file (<span className="font-mono text-slate-800">f.seek/read + q / 0.001</span>) without{' '}
          <span className="font-mono text-slate-800">mmap</span> overhead keeps active working RAM well under 1 GB even at
          1T scale.
        </p>
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Control Column */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-5">
          <div>
            <label htmlFor="scale-preset-select" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">
              Model Scale Preset
            </label>
            <select
              id="scale-preset-select"
              value={config.scaleId}
              onChange={(e) => onChange({ ...config, scaleId: e.target.value as ScalePresetId })}
              className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-900 focus:border-blue-600 focus:outline-none"
            >
              {SCALE_IDS.map((id) => (
                <option key={id} value={id}>
                  {MODEL_SCALE_PRESETS[id].label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="quant-regime-select" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">
              Compression / Storage Regime
            </label>
            <select
              id="quant-regime-select"
              value={config.quantId}
              onChange={(e) => onChange({ ...config, quantId: e.target.value as QuantizationRegimeId })}
              className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-900 focus:border-blue-600 focus:outline-none"
            >
              {QUANT_IDS.map((id) => (
                <option key={id} value={id}>
                  {QUANTIZATION_REGIMES[id].name}
                </option>
              ))}
            </select>
            <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">{selectedQuant.summary}</p>
          </div>

          <div>
            <span className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">
              RAM Execution Model
            </span>
            <div className="grid grid-cols-2 gap-1 p-1 bg-slate-100 border border-slate-200 rounded-md">
              {(
                [
                  { id: 'raw-x8d-direct', label: 'Raw .x8D Direct (No mmap)' },
                  { id: 'full-ram-resident', label: 'Full RAM Load' },
                ] as { id: ExecutionModeId; label: string }[]
              ).map((mode) => {
                const active = config.executionMode === mode.id;
                return (
                  <button
                    key={mode.id}
                    type="button"
                    onClick={() => onChange({ ...config, executionMode: mode.id })}
                    className={`px-2 py-1.5 text-[11px] font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
                      active
                        ? 'bg-white text-slate-900 shadow-xs border border-slate-200'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {mode.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="space-y-4 pt-2 border-t border-slate-200">
            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="ncm-skip-slider" className="font-medium text-slate-700">
                  NCM Coarse Block Skip Rate
                </label>
                <span className="font-mono tabular-nums text-slate-900">
                  {Math.round(config.ncmCoarseBypassRate * 100)}%
                </span>
              </div>
              <input
                id="ncm-skip-slider"
                type="range"
                min={0}
                max={85}
                step={5}
                value={Math.round(config.ncmCoarseBypassRate * 100)}
                onChange={(e) =>
                  onChange({ ...config, ncmCoarseBypassRate: Number(e.target.value) / 100 })
                }
                className="w-full accent-blue-600"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="context-bytes-slider" className="font-medium text-slate-700">
                  Sequence / Trajectory Window
                </label>
                <span className="font-mono tabular-nums text-slate-900">
                  {formatBytes(config.contextBytes)}
                </span>
              </div>
              <input
                id="context-bytes-slider"
                type="range"
                min={4096}
                max={131072}
                step={4096}
                value={config.contextBytes}
                onChange={(e) => onChange({ ...config, contextBytes: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="nodes-slider" className="font-medium text-slate-700">
                  Distributed FSDP / ZeRO-2 Nodes
                </label>
                <span className="font-mono tabular-nums text-slate-900">{config.distributedNodes} nodes</span>
              </div>
              <input
                id="nodes-slider"
                type="range"
                min={1}
                max={32}
                step={1}
                value={config.distributedNodes}
                onChange={(e) => onChange({ ...config, distributedNodes: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </div>
          </div>
        </div>

        {/* Right Telemetry & Chart Stage */}
        <div className="lg:col-span-8 space-y-5">
          {/* Live Telemetry Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 bg-white border border-slate-200 rounded-lg divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">Disk Container</div>
              <div className="mt-1 text-2xl font-mono font-semibold tabular-nums text-slate-900" data-testid="metric-disk-size">
                {formatBytes(result.diskBytes)}
              </div>
              <div className="mt-1 text-xs font-mono text-slate-500 tabular-nums">
                {result.effectiveBitsPerWeight.toFixed(3)} bpw · {result.compressionRatioVsFp16.toFixed(1)}x vs BF16
              </div>
            </div>

            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">Active Working RAM</div>
              <div className="mt-1 text-2xl font-mono font-semibold tabular-nums text-slate-900" data-testid="metric-active-ram">
                {formatBytes(result.activeWorkingSetRamBytes)}
              </div>
              <div className="mt-1 text-xs font-mono tabular-nums">
                {result.activeWorkingSetRamBytes <= 1024 * 1024 * 1024 ? (
                  <span className="text-emerald-700">
                    ● &lt;1 GB Phone/CPU Target (
                    {config.executionMode === 'raw-x8d-direct'
                      ? 'Direct f.seek/read RSS'
                      : config.executionMode === 'mmap-latent-file'
                        ? 'mmap RSS'
                        : 'RAM'}
                    )
                  </span>
                ) : (
                  <span className="text-rose-700">
                    ▲ Exceeds 1 GB Edge Ceiling
                  </span>
                )}
              </div>
            </div>

            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">Peak ALU Tile DRAM</div>
              <div className="mt-1 text-2xl font-mono font-semibold tabular-nums text-slate-900">
                {formatBytes(result.peakAluTileDramBytes)}
              </div>
              <div className="mt-1 text-xs font-mono text-slate-500 tabular-nums">
                1-layer dequant + FP32 tile
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 bg-white border border-slate-200 rounded-lg divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">Latent State File</div>
              <div className="mt-1 text-lg font-mono font-semibold tabular-nums text-slate-900">
                {formatBytes(result.latentStateFileBytes)}
              </div>
              <div className="mt-0.5 text-xs text-slate-500">KDA W + MLA KV + NCM grid</div>
            </div>

            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">Per-Node FSDP Shard</div>
              <div className="mt-1 text-lg font-mono font-semibold tabular-nums text-slate-900">
                {formatBytes(result.perNodeWeightShardBytes + result.perNodeOptimizerStateBytes)}
              </div>
              <div className="mt-0.5 text-xs text-slate-500">
                Weights + Muon/AdamW states / {config.distributedNodes}
              </div>
            </div>

            <div className="p-4">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">NCM Compute Ratio</div>
              <div className="mt-1 text-lg font-mono font-semibold tabular-nums text-slate-900">
                {(result.ncmEffectiveComputeFactor * 100).toFixed(1)}%
              </div>
              <div className="mt-0.5 text-xs font-mono tabular-nums">
                {result.forwardPassViable ? (
                  <span className="text-emerald-700">● Forward Pass Valid</span>
                ) : (
                  <span className="text-rose-700">✖ Forward Pass Invalid</span>
                )}
              </div>
            </div>
          </div>

          {/* Audit warnings */}
          {result.warnings.length > 0 && (
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-2">
              {result.warnings.map((warn, i) => (
                <div key={i} className="text-xs text-slate-700 leading-relaxed flex items-start gap-2">
                  <span className="font-mono font-semibold text-amber-700 shrink-0">NOTE:</span>
                  <span>{warn}</span>
                </div>
              ))}
            </div>
          )}

          {/* Comparison chart across all 5 regimes */}
          <div className="bg-white border border-slate-200 rounded-lg p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
              <h3 className="text-sm font-semibold text-slate-900">
                Regime Comparison at {formatParams(result.params.totalParams)} Scale (Log Scale Disk vs. Active RAM)
              </h3>
              <div className="flex items-center gap-4 text-xs font-mono text-slate-500">
                <span>■ Disk Size</span>
                <span className="text-cyan-700">■ Active RAM</span>
              </div>
            </div>

            <div className="mt-4 space-y-3">
              {comparisonRows.map(({ qId, spec, sim }) => {
                // Normalized log10 scale above 16 KB baseline so 50 KB vs 7 MB vs 98 MB separate crisply
                const logFloor = 4; // 10^4 = ~10 KB
                const logCeil = Math.max(logFloor + 1, Math.log10(Math.max(maxDiskBytes, 1024 * 1024)));
                const normPct = (bytes: number) => {
                  const v = Math.log10(Math.max(bytes, 10000));
                  return Math.min(100, Math.max(3, ((v - logFloor) / (logCeil - logFloor)) * 100));
                };
                const diskPct = normPct(sim.diskBytes);
                const ramPct = normPct(sim.activeWorkingSetRamBytes);
                const selected = qId === config.quantId;
                const statusLabel =
                  spec.auditStatus === 'broken'
                    ? '✖ Broken (#57)'
                    : spec.auditStatus === 'trap'
                      ? '▲ 8-bit Trap (#58)'
                      : '● Valid';
                const statusColor =
                  spec.auditStatus === 'broken'
                    ? 'text-rose-700'
                    : spec.auditStatus === 'trap'
                      ? 'text-amber-700'
                      : 'text-emerald-700';

                return (
                  <button
                    key={qId}
                    type="button"
                    onClick={() => onChange({ ...config, quantId: qId })}
                    className={`w-full text-left p-3 rounded border transition-colors cursor-pointer ${
                      selected
                        ? 'bg-blue-50/50 border-blue-600'
                        : 'bg-white border-slate-200 hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs mb-2">
                      <span className="font-semibold text-slate-900">{spec.shortLabel}</span>
                      <span className={`font-mono font-medium ${statusColor}`}>{statusLabel}</span>
                    </div>
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-2 bg-slate-100 rounded-xs overflow-hidden">
                          <div className="h-full rounded-xs bg-slate-800 transition-all" style={{ width: `${diskPct}%` }} />
                        </div>
                        <span className="w-28 text-right text-[11px] font-mono tabular-nums text-slate-700 shrink-0">
                          Disk: {formatBytes(sim.diskBytes)}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-2 bg-slate-100 rounded-xs overflow-hidden">
                          <div className="h-full rounded-xs bg-cyan-600 transition-all" style={{ width: `${ramPct}%` }} />
                        </div>
                        <span className="w-28 text-right text-[11px] font-mono tabular-nums text-cyan-800 shrink-0">
                          RAM: {formatBytes(sim.activeWorkingSetRamBytes)}
                        </span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
