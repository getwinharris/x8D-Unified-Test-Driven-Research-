import React, { useCallback, useEffect, useState } from 'react';
import { Cpu, Loader2, RefreshCw } from 'lucide-react';
import {
  NEEDLE3_TENSOR_SPECS,
  Needle3BenchmarkReport,
  Needle3TensorId,
} from '../lib/needle3Types';

const TENSOR_OPTIONS = Object.values(NEEDLE3_TENSOR_SPECS);

export const Needle3BenchmarkPanel: React.FC = () => {
  const [tensorId, setTensorId] = useState<Needle3TensorId>('self_attn_out_proj');
  const [report, setReport] = useState<Needle3BenchmarkReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchBenchmark = useCallback(async (targetTensor: Needle3TensorId, forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({ tensor: targetTensor });
      if (forceRefresh) qs.set('refresh', '1');
      const resp = await fetch(`/api/bench/needle3?${qs.toString()}`);
      const data = await resp.json();
      if (!resp.ok) {
        setError(data?.error || `HTTP ${resp.status} running needle3 benchmark.`);
        return;
      }
      setReport(data as Needle3BenchmarkReport);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Network error calling /api/bench/needle3.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchBenchmark(tensorId, false);
  }, [fetchBenchmark, tensorId]);

  const x8dUnpadded016 = report?.rows.find((r) => r.methodId === 'x8d-unpadded-016');
  const x8dUnpadded008 = report?.rows.find((r) => r.methodId === 'x8d-unpadded-008');
  const cactus = report?.rows.find((r) => r.methodId === 'cactus-official-cact');

  return (
    <div className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden" data-testid="needle3-real-bench">
      <div className="px-4 py-3.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-blue-600" />
            <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-900">
              Real Pretrained HuggingFace Comparison: Cactus-Compute/needle3 (121.02M FP16 vs .cact vs INT4/INT8 vs x8D)
            </h3>
          </div>
          <p className="text-[11px] text-slate-600 mt-0.5">
            Reads real weights directly via <span className="font-mono">fs.open/read</span> (no mmap) and tests{' '}
            <strong className="text-slate-800">parameter-to-parameter w×0.001</strong> against FP16, INT8, INT4, Cactus{' '}
            <span className="font-mono">.cact</span>, and Kronecker PTQ vs #56 QAT.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 min-w-0 w-full sm:w-auto">
          <select
            aria-label="Needle 3 tensor selector"
            data-testid="needle3-tensor-select"
            value={tensorId}
            onChange={(e) => setTensorId(e.target.value as Needle3TensorId)}
            className="w-full sm:w-auto min-w-0 max-w-full rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-mono text-slate-900 focus:border-blue-600 focus:outline-none"
          >
            {TENSOR_OPTIONS.map((spec) => (
              <option key={spec.id} value={spec.id}>
                {spec.label}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => fetchBenchmark(tensorId, true)}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-60 transition-colors cursor-pointer"
          >
            {loading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Measuring...</span>
              </>
            ) : (
              <>
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Re-Run Live Eval</span>
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div role="alert" className="p-4 text-xs bg-rose-50 text-rose-800 border-b border-rose-200">
          <strong className="font-mono">BENCHMARK ERROR: </strong>
          {error}
        </div>
      )}

      {report && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 p-4 border-b border-slate-200 bg-white text-xs">
            <div className="rounded border border-slate-200 bg-slate-50 p-3">
              <div className="text-[10px] font-mono uppercase text-slate-500">HF FP16 Safetensors</div>
              <div className="mt-0.5 text-sm font-mono font-bold text-slate-900 tabular-nums">
                {(report.safetensorsBytes / (1024 * 1024)).toFixed(2)} MB FP16
              </div>
              <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                {report.totalParameters.toLocaleString()} params
              </div>
            </div>

            <div className="rounded border border-slate-200 bg-slate-50 p-3">
              <div className="text-[10px] font-mono uppercase text-slate-500">Official needle3.cact</div>
              <div className="mt-0.5 text-sm font-mono font-bold text-slate-900 tabular-nums">
                {(report.officialCactBytes / (1024 * 1024)).toFixed(2)} MB ({report.officialCactBpw.toFixed(3)} bpw)
              </div>
              <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                Cactus SQNR: {cactus ? `${cactus.sqnrDb.toFixed(2)} dB` : '—'}
              </div>
            </div>

            <div className="rounded border border-blue-200 bg-blue-50/60 p-3">
              <div className="text-[10px] font-mono uppercase text-blue-700">x8D Unpadded 0.001× FP16</div>
              <div className="mt-0.5 text-sm font-mono font-bold text-blue-900 tabular-nums" data-testid="needle3-unpadded-016">
                {x8dUnpadded016 ? `${x8dUnpadded016.projectedNeedle3Mb.toFixed(5)} MB (${x8dUnpadded016.bpw.toFixed(3)} bpw)` : '—'}
              </div>
              <div className="text-[11px] font-mono text-emerald-700 font-semibold mt-0.5">
                230.83→0.23083 MB · Zero padding
              </div>
            </div>

            <div className="rounded border border-emerald-200 bg-emerald-50/50 p-3">
              <div className="text-[10px] font-mono uppercase text-emerald-800">x8D Unpadded 8-bit→0.008b</div>
              <div className="mt-0.5 text-sm font-mono font-bold text-emerald-950 tabular-nums" data-testid="needle3-unpadded-008">
                {x8dUnpadded008 ? `${x8dUnpadded008.projectedNeedle3Mb.toFixed(5)} MB (${x8dUnpadded008.bpw.toFixed(3)} bpw)` : '—'}
              </div>
              <div className="text-[11px] font-mono text-slate-600 mt-0.5">
                Bit-exact roundtrip · {report.benchmarkDurationMs} ms
              </div>
            </div>
          </div>

          <div className="px-4 py-2 bg-slate-50/70 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-600">
            <span>
              Tensor: <strong className="text-slate-900">{report.tensor.safetensorsKey}</strong> · range [
              {report.tensorStats.min}, {report.tensorStats.max}] · RMS {report.tensorStats.rms} · |w|&lt;0.01:{' '}
              {report.tensorStats.nearZeroPct}%
            </span>
            <span className="font-semibold text-blue-700">Direct fs.open/read (No mmap)</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                  <th className="py-2.5 px-4">Quantization Regime</th>
                  <th className="py-2.5 px-3 text-right">BPW</th>
                  <th className="py-2.5 px-3 text-right">Proj. 121M</th>
                  <th className="py-2.5 px-3 text-right">Weight Cos</th>
                  <th className="py-2.5 px-3 text-right">SQNR</th>
                  <th className="py-2.5 px-3 text-right">Act Cos</th>
                  <th className="py-2.5 px-3 text-right">Logit KL</th>
                  <th className="py-2.5 px-4">Empirical Finding</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-xs">
                {report.rows.map((row) => {
                  const isX8d = row.family === 'x8d';
                  const isCactus = row.family === 'cactus';
                  const isLowCos = row.cosineSimilarity < 0.35;
                  return (
                    <tr
                      key={row.methodId}
                      className={
                        isX8d
                          ? isLowCos
                            ? 'bg-amber-50/50 align-top'
                            : 'bg-blue-50/50 align-top'
                          : isCactus
                            ? 'bg-emerald-50/40 align-top'
                            : 'hover:bg-slate-50/70 align-top'
                      }
                    >
                      <td className="py-2.5 px-4 font-semibold text-slate-900 whitespace-nowrap">
                        {row.label}
                      </td>
                      <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                        {row.bpw.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                        {row.projectedNeedle3Mb < 1
                          ? `${row.projectedNeedle3Mb.toFixed(4)} MB`
                          : `${row.projectedNeedle3Mb.toFixed(2)} MB`}
                      </td>
                      <td
                        className={`py-2.5 px-3 font-mono tabular-nums text-right font-semibold whitespace-nowrap ${
                          row.cosineSimilarity >= 0.85
                            ? 'text-emerald-700'
                            : row.cosineSimilarity >= 0.6
                              ? 'text-blue-700'
                              : 'text-rose-700'
                        }`}
                      >
                        {row.cosineSimilarity.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                        {row.sqnrDb > 90 ? 'Inf (Ref)' : `${row.sqnrDb.toFixed(2)} dB`}
                      </td>
                      <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                        {row.outputActCosine.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                        {row.logitKlDivergence.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-4 text-slate-600 leading-relaxed">{row.verdictNote}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-700 leading-relaxed">
            <strong className="font-semibold text-slate-900">Needle3 Verdict: </strong>
            {report.architectTakeaway}
          </div>
        </>
      )}
    </div>
  );
};
