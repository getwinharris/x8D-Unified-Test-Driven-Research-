import React, { useMemo, useState } from 'react';
import {
  ExecutionModeId,
  FootprintSimulationInput,
  MODEL_SCALE_PRESETS,
  PRECISION_FORMAT_BASELINE,
  ScalePresetId,
  computeUniqueSubByteArith,
  formatBytes,
  formatParams,
  simulateFootprint,
  simulateUnifiedX8DImage,
} from '../lib/architectureMath';
import { runSubBitComparisonBenchmark } from '../lib/quantBenchmarkEngine';
import { Needle3BenchmarkPanel } from './Needle3BenchmarkPanel';

interface SubBitFrontierSectionProps {
  config: FootprintSimulationInput;
  onChange: (next: FootprintSimulationInput) => void;
}

const SCALE_IDS: ScalePresetId[] = ['ncm-50m', 'ncm-150m', 'x8d-1b', 'x8d-1t'];

export const SubBitFrontierSection: React.FC<SubBitFrontierSectionProps> = ({ config, onChange }) => {
  const [attentionCodec, setAttentionCodec] = useState<'kron-factored' | 'kron-subbyte-0008'>('kron-subbyte-0008');
  const [kTerms, setKTerms] = useState<number>(1);
  const [kBits, setKBits] = useState<number>(4);

  const [inspectMode, setInspectMode] = useState<'io-byte' | 'model-param'>('model-param');
  const [sampleByte, setSampleByte] = useState<number>(65);
  const [sampleParam, setSampleParam] = useState<number>(121.57266);
  const [priorPct, setPriorPct] = useState<number>(99.98);

  const liveBench = useMemo(() => runSubBitComparisonBenchmark(), []);
  const footprint = useMemo(() => simulateFootprint(config), [config]);

  const x8dImage = useMemo(
    () =>
      simulateUnifiedX8DImage({
        scaleId: config.scaleId,
        attentionCodec,
        kroneckerTerms: kTerms,
        kroneckerBits: kBits,
        bytePriorAccuracy: priorPct / 100,
      }),
    [config.scaleId, attentionCodec, kTerms, kBits, priorPct]
  );

  const arithByte = useMemo(
    () =>
      computeUniqueSubByteArith({
        byteValue: sampleByte,
        streamLengthBytes: 100_000,
        priorMatchRate: priorPct / 100,
      }),
    [sampleByte, priorPct]
  );

  const charPreview =
    sampleByte >= 32 && sampleByte <= 126
      ? `'${String.fromCharCode(sampleByte)}'`
      : `0x${sampleByte.toString(16).toUpperCase().padStart(2, '0')}`;

  return (
    <section id="subbit-frontier" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          03. 0.008 bpw (0.001 Sub-Byte) .x8D Storage &amp; Direct f.seek/read Serving
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          How Raw .x8D Reaches 0.008 Bits Per 8-Bit Without mmap or -1 Ternary
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Standard compute sits on 8-bit hardware boundaries, so input and output bytes map between{' '}
          <span className="font-mono text-slate-800">0..255</span> and{' '}
          <span className="font-mono text-slate-800">0.000..0.255</span>, while stored weights compress toward{' '}
          <strong className="font-semibold text-slate-900">0.008 bpw</strong> via Kronecker decomposition + QAT (#56) +{' '}
          <span className="font-mono text-slate-800">x8d_arith</span> (#60/#61). At runtime, plain{' '}
          <span className="font-mono text-slate-800">open(&quot;rb&quot;) + f.seek/f.read</span> streams active spans
          directly into L1/L2 CPU tiles with zero OS <span className="font-mono text-slate-800">mmap</span>.
        </p>
      </div>

      {/* Consolidated .x8D Disk + Direct f.seek/read Footprint & w*0.001 Verifier */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-7 min-w-0 bg-white border border-slate-200 rounded-lg p-5 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">.x8D Disk &amp; Direct f.seek/read RAM Simulator</h3>
              <p className="text-xs text-slate-600 mt-0.5">
                Compare 50M prototype up to 1.00T phone/CPU target (&lt;1 GB RAM, no mmap).
              </p>
            </div>
            <span className="font-mono text-xs text-emerald-700 font-semibold">0.008 bpw · 0.001 B/w</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label htmlFor="x8d-scale-select" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-1">
                Model Scale
              </label>
              <select
                id="x8d-scale-select"
                value={config.scaleId}
                onChange={(e) => onChange({ ...config, scaleId: e.target.value as ScalePresetId })}
                className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-900 focus:border-blue-600 focus:outline-none"
              >
                {SCALE_IDS.map((id) => (
                  <option key={id} value={id}>
                    {MODEL_SCALE_PRESETS[id].id.toUpperCase()} ({formatParams(simulateFootprint({ ...config, scaleId: id }).params.totalParams)})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="x8d-exec-mode" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-1">
                Runtime Streaming
              </label>
              <select
                id="x8d-exec-mode"
                value={config.executionMode}
                onChange={(e) => onChange({ ...config, executionMode: e.target.value as ExecutionModeId })}
                className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-900 focus:border-blue-600 focus:outline-none"
              >
                <option value="raw-x8d-direct">Raw .x8D Direct (f.seek/read, No mmap)</option>
                <option value="full-ram-resident">Full RAM Resident</option>
              </select>
            </div>

            <div>
              <label htmlFor="x8d-attn-codec" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-1">
                Weight Packing Path
              </label>
              <select
                id="x8d-attn-codec"
                value={attentionCodec}
                onChange={(e) => setAttentionCodec(e.target.value as 'kron-factored' | 'kron-subbyte-0008')}
                className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-900 focus:border-blue-600 focus:outline-none"
              >
                <option value="kron-subbyte-0008">Kronecker + x8d_arith (0.008 bpw)</option>
                <option value="kron-factored">Kronecker Low-Rank PTQ Check</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 border border-slate-200 bg-slate-50 rounded-md divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
            <div className="p-3">
              <div className="text-[11px] font-mono text-slate-500">.x8D On Disk</div>
              <div className="mt-0.5 text-base font-mono font-bold text-slate-900 tabular-nums" data-testid="x8d-disk-bytes">
                {formatBytes(x8dImage.unifiedX8DBytes)}
              </div>
              <div className="text-[11px] font-mono text-emerald-700" data-testid="x8d-compression">
                {x8dImage.compressionFactorVsBf16.toFixed(0)}× vs BF16
              </div>
            </div>

            <div className="p-3">
              <div className="text-[11px] font-mono text-slate-500">Active Working RAM</div>
              <div className="mt-0.5 text-base font-mono font-bold text-slate-900 tabular-nums">
                {formatBytes(footprint.activeWorkingSetRamBytes)}
              </div>
              <div className="text-[11px] font-mono text-slate-600">
                {config.executionMode === 'raw-x8d-direct' ? 'f.seek/read RSS' : 'Resident RAM'}
              </div>
            </div>

            <div className="p-3">
              <div className="text-[11px] font-mono text-slate-500">ALU Tile SRAM/DRAM</div>
              <div className="mt-0.5 text-base font-mono font-bold text-slate-900 tabular-nums">
                {formatBytes(footprint.peakAluTileDramBytes)}
              </div>
              <div className="text-[11px] font-mono text-slate-600">1 active layer slice</div>
            </div>

            <div className="p-3">
              <div className="text-[11px] font-mono text-slate-500">Tail Seek Directory</div>
              <div className="mt-0.5 text-base font-mono font-bold text-slate-900 tabular-nums">
                {formatBytes(x8dImage.framingOverheadBytes)}
              </div>
              <div className="text-[11px] font-mono text-slate-600">{x8dImage.totalTensorsPacked} spans (#52)</div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <div className="flex items-center justify-between mb-1">
                <label htmlFor="kron-k-slider" className="font-medium text-slate-700">
                  Kronecker Factor Terms (K)
                </label>
                <span className="font-mono tabular-nums text-slate-900">K = {kTerms}</span>
              </div>
              <input
                id="kron-k-slider"
                type="range"
                min={1}
                max={4}
                step={1}
                value={kTerms}
                onChange={(e) => setKTerms(Number(e.target.value))}
                className="w-full accent-blue-600"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label htmlFor="kron-bits-slider" className="font-medium text-slate-700">
                  Factor Bit Precision
                </label>
                <span className="font-mono tabular-nums text-slate-900">{kBits}-bit</span>
              </div>
              <input
                id="kron-bits-slider"
                type="range"
                min={2}
                max={4}
                step={1}
                value={kBits}
                onChange={(e) => setKBits(Number(e.target.value))}
                className="w-full accent-blue-600"
              />
            </div>
          </div>
        </div>

        {/* w * 0.001 Parameter-to-Parameter & 8-bit Byte Verifier */}
        <div className="lg:col-span-5 min-w-0 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="pb-3 border-b border-slate-200">
            <h3 className="text-sm font-semibold text-slate-900">Raw .x8D Direct Verifier (× / ÷ 0.001)</h3>
            <p className="mt-1 text-xs text-slate-600">
              Inspects continuous weights (<span className="font-mono">w × 0.001</span>, e.g.{' '}
              <span className="font-mono">121.57266 → 0.12157266</span>) vs. 8-bit I/O bytes (
              <span className="font-mono">0..255 → 0.000..0.255</span>) + <span className="font-mono">x8d_arith</span>.
            </p>
            <div className="mt-2.5 grid grid-cols-2 gap-1.5">
              <button
                type="button"
                onClick={() => setInspectMode('model-param')}
                className={`rounded border px-2 py-1 text-[11px] font-mono cursor-pointer ${
                  inspectMode === 'model-param'
                    ? 'border-blue-600 bg-blue-50 text-blue-800 font-semibold'
                    : 'border-slate-200 bg-white text-slate-700'
                }`}
              >
                Param w×0.001
              </button>
              <button
                type="button"
                onClick={() => setInspectMode('io-byte')}
                className={`rounded border px-2 py-1 text-[11px] font-mono cursor-pointer ${
                  inspectMode === 'io-byte'
                    ? 'border-blue-600 bg-blue-50 text-blue-800 font-semibold'
                    : 'border-slate-200 bg-white text-slate-700'
                }`}
              >
                8-bit I/O Byte
              </button>
            </div>
          </div>

          {inspectMode === 'model-param' ? (
            <div className="space-y-3">
              <div>
                <label htmlFor="param-float-input" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-1">
                  Pretrained Parameter (Float)
                </label>
                <input
                  id="param-float-input"
                  data-testid="param-float-input"
                  type="number"
                  step="any"
                  value={sampleParam}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setSampleParam(Number.isFinite(v) ? v : 0);
                  }}
                  className="w-full rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-mono text-slate-900 focus:border-blue-600 focus:outline-none"
                />
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-md p-3 space-y-1.5 text-xs font-mono">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">w × 0.001 Sub-Byte</span>
                  <span className="font-bold text-blue-700 tabular-nums" data-testid="param-subbyte-val">
                    {(sampleParam * 0.001).toFixed(8)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">w_sub / 0.001 Decode</span>
                  <span className="font-semibold text-emerald-700 tabular-nums">
                    {((sampleParam * 0.001) / 0.001).toFixed(5)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">0.001-Step Threshold</span>
                  <span className="text-slate-800 tabular-nums">
                    {(Math.round((sampleParam * 0.001) / 0.001) * 0.001).toFixed(3)} →{' '}
                    {((Math.round((sampleParam * 0.001) / 0.001) * 0.001) / 0.001).toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="unique-byte-input" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-1">
                  I/O Byte (0–255)
                </label>
                <input
                  id="unique-byte-input"
                  type="number"
                  min={0}
                  max={255}
                  value={sampleByte}
                  onChange={(e) => setSampleByte(Math.max(0, Math.min(255, Number(e.target.value) || 0)))}
                  className="w-full rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-mono text-slate-900 focus:border-blue-600 focus:outline-none"
                />
              </div>
              <div className="bg-slate-50 border border-slate-200 rounded-md px-3 py-1.5 flex flex-col justify-center">
                <div className="text-[11px] font-mono text-slate-500">I/O Quanta Map</div>
                <div className="text-xs font-mono font-semibold text-slate-900" data-testid="unique-byte-map">
                  {charPreview} ({arithByte.byteValue}) → {arithByte.subByteScaledValue.toFixed(3)}
                </div>
              </div>
            </div>
          )}

          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <label htmlFor="prior-match-slider" className="font-medium text-slate-700">
                x8d_arith Predictor Hit Rate
              </label>
              <span className="font-mono tabular-nums text-slate-900">{priorPct.toFixed(2)}%</span>
            </div>
            <input
              id="prior-match-slider"
              type="range"
              min={90}
              max={99.99}
              step={0.01}
              value={priorPct}
              onChange={(e) => setPriorPct(Number(e.target.value))}
              className="w-full accent-blue-600"
            />
          </div>

          <div className="pt-3 border-t border-slate-200 space-y-1.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-600">x8d_arith Frame (100 KB Source)</span>
              <span className="font-mono font-bold text-base tabular-nums text-slate-900" data-testid="arith-disk-bytes">
                {formatBytes(arithByte.arithDiskBytes)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600">Effective Bit / Byte Rate</span>
              <span className="font-mono font-semibold tabular-nums text-emerald-700" data-testid="arith-bpw">
                {arithByte.arithBitsPerByte.toFixed(4)} bits ({arithByte.arithBytesPerByte.toFixed(5)} B/elem)
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Precision Format Baseline Table */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between gap-4">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
            Standard Precision Baselines vs. Unified .x8D Target (0.008 bpw · Zero -1 Ternary)
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                <th className="py-2.5 px-4">Format</th>
                <th className="py-2.5 px-3">Bits / Weight</th>
                <th className="py-2.5 px-3">Bytes / Weight</th>
                <th className="py-2.5 px-4">Role in Storage vs. 8-Bit I/O</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-xs">
              {PRECISION_FORMAT_BASELINE.map((row, idx) => {
                const isX8d = idx === PRECISION_FORMAT_BASELINE.length - 1;
                return (
                  <tr
                    key={row.format}
                    className={isX8d ? 'bg-blue-50/50 align-top' : 'hover:bg-slate-50/70 align-top'}
                  >
                    <td className="py-2.5 px-4 font-semibold text-slate-900 whitespace-nowrap">{row.format}</td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-slate-900 whitespace-nowrap">
                      {row.bpwLabel}
                    </td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-slate-900 whitespace-nowrap">
                      {row.bytesPerParamLabel}
                    </td>
                    <td className="py-2.5 px-4 text-slate-600 leading-relaxed">{row.description}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Measured Side-by-Side Quantization, Perplexity & Snake Benchmark */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden" data-testid="live-perplexity-bench">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
              Side-by-Side Quantization, Byte Perplexity &amp; Snake Eval ({liveBench.matrixShape})
            </h3>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Compares BF16, Regular INT8, Regular INT4, x8D Kronecker + x8d_arith (0.008 bpw QAT), and Broken #57 block-mean across{' '}
              {liveBench.totalWeightsEvaluated.toLocaleString()} weights and {liveBench.evalStreamBytes} bytes.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-600 font-semibold">npm run bench:subbit</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                <th className="py-2.5 px-4">Quantization Method</th>
                <th className="py-2.5 px-3 text-right">BPW</th>
                <th className="py-2.5 px-3 text-right">Cosine</th>
                <th className="py-2.5 px-3 text-right">Byte PPL</th>
                <th className="py-2.5 px-3 text-right">ΔPPL vs BF16</th>
                <th className="py-2.5 px-3 text-right">Top-1 Acc</th>
                <th className="py-2.5 px-3 text-right">Snake Score</th>
                <th className="py-2.5 px-4">On-Disk / Tile RAM Path</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-xs">
              {liveBench.rows.map((row) => {
                const isBroken = row.methodId === 'legacy-blockmean-57';
                const isX8d = row.methodId.startsWith('x8d-');
                return (
                  <tr
                    key={row.methodId}
                    className={
                      isBroken
                        ? 'bg-rose-50/50 align-top'
                        : isX8d
                          ? 'bg-blue-50/50 align-top'
                          : 'hover:bg-slate-50/70 align-top'
                    }
                  >
                    <td className="py-2.5 px-4 font-semibold text-slate-900 whitespace-nowrap">{row.label}</td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                      {row.bpw.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-slate-900 whitespace-nowrap">
                      {row.cosineSimilarity.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                      {row.bytePerplexity.toFixed(2)}
                    </td>
                    <td
                      className={`py-2.5 px-3 font-mono tabular-nums text-right whitespace-nowrap ${
                        row.pplDeltaVsBf16 <= 1.5 ? 'text-emerald-700' : isBroken ? 'text-rose-700 font-semibold' : 'text-amber-700'
                      }`}
                    >
                      {row.pplDeltaVsBf16 >= 0 ? `+${row.pplDeltaVsBf16.toFixed(2)}` : row.pplDeltaVsBf16.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                      {row.headAccuracyPct.toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                      {row.snakeFoodScore120Steps} pts
                    </td>
                    <td className="py-2.5 px-4 text-slate-600 leading-relaxed">{row.onDiskTileRamNote}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-600 leading-relaxed">
          <strong className="font-semibold text-slate-900">Key Engineering Distinction: </strong>
          {liveBench.shannonWallInsight}
        </div>
      </div>

      <Needle3BenchmarkPanel />
    </section>
  );
};
