import React, { useCallback, useEffect, useState } from 'react';
import { CheckCircle2, FileCode2, FlaskConical, HardDrive, Layers, Play, RefreshCw, XCircle } from 'lucide-react';
import type { X8dResearchSuiteReport } from '../lib/researchSuiteTypes';

type LabTabId = 'codec' | 'ncm' | 'xat' | 'mp4' | 'hypotheses';

interface LiveCodecTestResult {
  mode: string;
  rawBytes: number;
  physicalX8dBytes: number;
  payloadSpanBytes: number;
  codecUsed: string;
  compressionRatio: number;
  storedBitsPerByte: number;
  sha256Original: string;
  sha256Decoded: string;
  sha256ExactMatch: boolean;
  crc32Hex: string;
  encodeMs: number;
  decodeMs: number;
  decodedPreview: string;
}

function fmtBytes(n: number): string {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(2)} KB`;
  return `${n.toLocaleString()} B`;
}

export const ResearchBriefLabSection: React.FC = () => {
  const [activeTab, setActiveTab] = useState<LabTabId>('codec');
  const [report, setReport] = useState<X8dResearchSuiteReport | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [running, setRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Interactive Step A->B->C Live Codec Verifier state
  const [codecMode, setCodecMode] = useState<'text' | 'vocab256' | 'png' | 'wav' | 'random'>('text');
  const [codecInputText, setCodecInputText] = useState<string>(
    'வணக்கம் உலகம் — x8D 0.000..0.255 Unified Byte Loop'
  );
  const [codecTesting, setCodecTesting] = useState<boolean>(false);
  const [codecLiveResult, setCodecLiveResult] = useState<LiveCodecTestResult | null>(null);
  const [codecLiveError, setCodecLiveError] = useState<string | null>(null);

  const fetchCachedReport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch('/api/research/suite');
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.error || `HTTP ${resp.status}`);
      setReport(data as X8dResearchSuiteReport);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Could not load empirical research suite report.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchCachedReport();
  }, [fetchCachedReport]);

  const handleRerunSuite = async () => {
    setRunning(true);
    setError(null);
    try {
      const resp = await fetch('/api/research/suite/run', { method: 'POST' });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.error || `HTTP ${resp.status}`);
      setReport(data as X8dResearchSuiteReport);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed re-running empirical research suite.');
    } finally {
      setRunning(false);
    }
  };

  const handleRunInteractiveCodecTest = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setCodecTesting(true);
    setCodecLiveError(null);
    try {
      const resp = await fetch('/api/research/codec-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: codecMode, text: codecInputText }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.error || `HTTP ${resp.status}`);
      setCodecLiveResult(data as LiveCodecTestResult);
    } catch (err: unknown) {
      setCodecLiveError(err instanceof Error ? err.message : 'Interactive .x8D roundtrip failed.');
    } finally {
      setCodecTesting(false);
    }
  };

  const tabs: Array<{ id: LabTabId; label: string }> = [
    { id: 'codec', label: '01. Reversible Codec & 4 Regimes' },
    { id: 'ncm', label: '02. NCM RVQ & Stage 1–5 Eval' },
    { id: 'xat', label: '03. xAt = 65,536 Sequence' },
    { id: 'mp4', label: '04. MP4 Direct-Factor Exec' },
    { id: 'hypotheses', label: '05. Failed Hypotheses & Gates' },
  ];

  return (
    <section
      id="research-brief-lab"
      data-testid="research-brief-lab"
      className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200"
    >
      <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
        <div className="max-w-3xl">
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono uppercase tracking-wider text-slate-500">
            <span>04. Empirical Research Lab</span>
            <span aria-hidden="true">·</span>
            <span>Reproducible On-Disk Artifacts</span>
            {report && (
              <>
                <span aria-hidden="true">·</span>
                <span className="tabular-nums">Last Suite Run: {report.totalDurationMs} ms</span>
              </>
            )}
          </div>
          <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
            Measured Codec Bijection, NCM RVQ Ablation, xAt=65,536 Window &amp; MP4 Direct-Factor Execution
          </h2>
          <p className="mt-2 text-sm text-slate-600 leading-relaxed">
            Reproducible Node &amp; binary <span className="font-mono text-slate-800">.x8D v2</span> artifact experiments separating lossless I/O, model serialization,
            lossy Kronecker factorization, NCM hierarchical RVQ with nested dropout, 64 KiB sequence routing, and
            direct-from-factor execution without dense matrix materialization.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            data-testid="rerun-research-suite-btn"
            onClick={handleRerunSuite}
            disabled={running || loading}
            className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-60 transition-colors cursor-pointer whitespace-nowrap shrink-0"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${running ? 'animate-spin' : ''}`} />
            <span>{running ? 'Running Suite on Disk...' : 'Re-Run Full Suite'}</span>
          </button>
        </div>
      </div>

      {/* Single-line Segmented Experiment Tabs */}
      <div
        role="tablist"
        aria-label="Research Brief Empirical Experiments"
        className="mt-6 flex flex-wrap gap-1.5 p-1 bg-slate-100 rounded-lg border border-slate-200 w-full sm:w-fit max-w-full"
      >
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              data-testid={`lab-tab-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                isActive
                  ? 'bg-white text-slate-900 border border-slate-200 font-semibold'
                  : 'text-slate-600 hover:text-slate-900 border border-transparent'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {error && (
        <div className="mt-4 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-xs font-mono text-rose-800">
          Error loading research suite: {error}
        </div>
      )}

      {loading && !report ? (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-8 text-center text-xs font-mono text-slate-600">
          Executing &amp; loading physical <span className="font-mono">.x8D v2</span> experimental artifacts from disk...
        </div>
      ) : report ? (
        <div className="mt-6 space-y-6">
          {/* TAB 1: REVERSIBLE CODEC & 4 REGIMES */}
          {activeTab === 'codec' && (
            <div className="space-y-6" data-testid="lab-panel-codec">
              {/* Fixed-point milli-units vs Float64 vs Float32 trunc */}
              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                    Fixed-Point Integer Milli-Units vs. Float64 &amp; Float32 Roundtrip (0..255 Complete Alphabet)
                  </h3>
                  <span className="text-xs font-mono text-emerald-700 font-semibold">
                    {report.codecExperiment.allPayloadsBitExact ? '8/8 Artifacts SHA-256 Identical' : 'Mismatch Detected'}
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                        <th className="py-2.5 px-4">Mapping Regime</th>
                        <th className="py-2.5 px-3">Formula</th>
                        <th className="py-2.5 px-3 text-right">Tested</th>
                        <th className="py-2.5 px-3 text-right">Collisions</th>
                        <th className="py-2.5 px-3 text-right">Exact %</th>
                        <th className="py-2.5 px-4">Empirical Finding</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-xs">
                      {report.codecExperiment.fixedPointComparison.map((row) => (
                        <tr key={row.method} className={row.collisionCount > 0 ? 'bg-rose-50/40' : 'bg-white'}>
                          <td className="py-2.5 px-4 font-semibold text-slate-900 whitespace-nowrap">{row.method}</td>
                          <td className="py-2.5 px-3 font-mono text-[11px] text-slate-700 whitespace-nowrap">{row.formula}</td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800">{row.testedValues}</td>
                          <td
                            className={`py-2.5 px-3 font-mono tabular-nums text-right font-semibold ${
                              row.collisionCount === 0 ? 'text-emerald-700' : 'text-rose-700'
                            }`}
                          >
                            {row.collisionCount}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900">
                            {row.exactBijectionPct.toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-4 text-slate-600">{row.verdict}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Physical .x8D Step A -> B -> C across 8 Payloads */}
              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                      Step A → Step B → Step C Real Disk .x8D v2 Codec Verification (PNG, WAV, Tamil, 0..255, Random, xAt=65,536)
                    </h3>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      PNG Valid IHDR/IDAT/IEND: {report.codecExperiment.pngValidHeaderAndChunks ? 'Yes' : 'No'} · WAV Valid RIFF/PCM:{' '}
                      {report.codecExperiment.wavValidRiffHeader ? 'Yes' : 'No'} · Container Header+Tail = 64 B
                    </p>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                        <th className="py-2.5 px-4">Payload</th>
                        <th className="py-2.5 px-3 text-right">Raw B</th>
                        <th className="py-2.5 px-3 text-right">H (b/B)</th>
                        <th className="py-2.5 px-3 text-right">.x8D Disk B</th>
                        <th className="py-2.5 px-3 text-right">Stored bpb</th>
                        <th className="py-2.5 px-3 text-right">Ratio</th>
                        <th className="py-2.5 px-3">Codec</th>
                        <th className="py-2.5 px-3 text-right">SHA-256 Parity</th>
                        <th className="py-2.5 px-3 text-right">Enc / Dec ms</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-xs">
                      {report.codecExperiment.payloadAudits.map((row) => (
                        <tr key={row.payloadId} className="hover:bg-slate-50/70 align-top">
                          <td className="py-2.5 px-4">
                            <div className="font-semibold text-slate-900">{row.label}</div>
                            <div className="text-[11px] text-slate-500 mt-0.5 max-w-md">{row.honestNote}</div>
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                            {row.rawBytes.toLocaleString()}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-600 whitespace-nowrap">
                            {row.shannonEntropyBitsPerByte.toFixed(2)}
                          </td>
                          <td
                            className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-slate-900 whitespace-nowrap"
                            data-testid={`codec-x8d-bytes-${row.payloadId}`}
                          >
                            {row.physicalX8dBytes.toLocaleString()}
                          </td>
                          <td
                            className={`py-2.5 px-3 font-mono tabular-nums text-right font-semibold whitespace-nowrap ${
                              row.storedBitsPerByte > 8.0 ? 'text-amber-700' : 'text-blue-700'
                            }`}
                          >
                            {row.storedBitsPerByte.toFixed(3)}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                            {row.compressionRatio.toFixed(2)}×
                          </td>
                          <td className="py-2.5 px-3 font-mono text-[11px] text-slate-700 whitespace-nowrap">
                            {row.codecUsed}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right whitespace-nowrap">
                            {row.sha256ExactMatch && row.crc32Verified ? (
                              <span className="inline-flex items-center gap-1 text-emerald-700 font-semibold">
                                <CheckCircle2 className="w-3.5 h-3.5" /> 100% ({row.sha256Decoded.slice(0, 8)})
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-rose-700 font-semibold">
                                <XCircle className="w-3.5 h-3.5" /> Mismatch
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-600 whitespace-nowrap">
                            {row.encodeMs.toFixed(2)} / {row.decodeMs.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-700 leading-relaxed">
                  <strong className="font-semibold text-slate-900">Empirical Reality Check: </strong>
                  {report.codecExperiment.honestFinding}
                </div>
              </div>

              {/* 4 Separate Regimes Table */}
              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
                  <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                    Four Strictly Separated Storage &amp; Execution Regimes
                  </h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                        <th className="py-2.5 px-4">Regime</th>
                        <th className="py-2.5 px-3">Domain</th>
                        <th className="py-2.5 px-3">Measured Rate</th>
                        <th className="py-2.5 px-3">Lossless?</th>
                        <th className="py-2.5 px-4">Fidelity &amp; Scope Boundary</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-xs">
                      {report.codecExperiment.fourRegimes.map((reg) => (
                        <tr key={reg.regimeId} className="align-top">
                          <td className="py-2.5 px-4 font-semibold text-slate-900">{reg.label}</td>
                          <td className="py-2.5 px-3 text-slate-700">{reg.domain}</td>
                          <td className="py-2.5 px-3 font-mono text-blue-800 font-semibold">{reg.measuredRate}</td>
                          <td className="py-2.5 px-3 font-mono font-semibold whitespace-nowrap">
                            <span className={reg.exactLossless ? 'text-emerald-700' : 'text-amber-700'}>
                              {reg.exactLossless ? 'Lossless (SHA-256)' : 'Lossy / Compact'}
                            </span>
                          </td>
                          <td className="py-2.5 px-4 text-slate-600">
                            <div className="font-mono text-slate-800">{reg.reconstructionMetric}</div>
                            <div className="mt-0.5 text-[11px]">{reg.scopeBoundary}</div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Interactive Live Codec Test Strip */}
              <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
                  <div className="flex items-center gap-2">
                    <FileCode2 className="w-4 h-4 text-blue-700" />
                    <h3 className="text-sm font-semibold text-slate-900">
                      Interactive Live .x8D v2 Forward → Disk Size → Reload-Only Reverse Test
                    </h3>
                  </div>
                  <span className="text-xs font-mono text-slate-500">Writes &amp; Reloads From Disk</span>
                </div>

                <form onSubmit={handleRunInteractiveCodecTest} className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    {[
                      { id: 'text', label: 'Custom Text / Tamil UTF-8' },
                      { id: 'vocab256', label: 'All 0..255 Bytes' },
                      { id: 'png', label: 'Valid 32×32 PNG Binary' },
                      { id: 'wav', label: 'Valid 8 kHz WAV Audio' },
                      { id: 'random', label: '2 KB High-Entropy Random' },
                    ].map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setCodecMode(opt.id as typeof codecMode)}
                        className={`rounded border px-3 py-1.5 text-xs font-medium cursor-pointer ${
                          codecMode === opt.id
                            ? 'border-blue-600 bg-blue-50 text-blue-900 font-semibold'
                            : 'border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>

                  {codecMode === 'text' && (
                    <input
                      type="text"
                      aria-label="Custom byte payload to encode into .x8D v2"
                      value={codecInputText}
                      onChange={(e) => setCodecInputText(e.target.value)}
                      className="w-full rounded border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-900 focus:border-blue-600 focus:outline-none"
                    />
                  )}

                  <div className="flex items-center gap-3">
                    <button
                      type="submit"
                      data-testid="run-live-codec-btn"
                      disabled={codecTesting}
                      className="inline-flex items-center gap-2 rounded bg-blue-700 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-800 disabled:opacity-60 cursor-pointer"
                    >
                      <Play className="w-3.5 h-3.5" />
                      <span>{codecTesting ? 'Encoding & Reloading...' : 'Write .x8D & Verify SHA-256 Roundtrip'}</span>
                    </button>
                  </div>
                </form>

                {codecLiveError && (
                  <div className="rounded border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-mono text-rose-800">
                    {codecLiveError}
                  </div>
                )}

                {codecLiveResult && (
                  <div
                    className="grid grid-cols-1 md:grid-cols-4 gap-3 rounded border border-slate-200 bg-slate-50 p-3.5 text-xs font-mono"
                    data-testid="live-codec-result"
                  >
                    <div>
                      <div className="text-[10px] uppercase text-slate-500">Raw → Physical .x8D</div>
                      <div className="mt-0.5 font-semibold text-slate-900 tabular-nums">
                        {codecLiveResult.rawBytes} B → {codecLiveResult.physicalX8dBytes} B ({codecLiveResult.codecUsed})
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase text-slate-500">Rate / Ratio</div>
                      <div className="mt-0.5 font-semibold text-blue-800 tabular-nums">
                        {codecLiveResult.storedBitsPerByte.toFixed(3)} bpb · {codecLiveResult.compressionRatio}×
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase text-slate-500">SHA-256 &amp; CRC32</div>
                      <div className="mt-0.5 font-semibold text-emerald-700 tabular-nums">
                        {codecLiveResult.sha256ExactMatch ? 'BIT-EXACT' : 'FAIL'} · {codecLiveResult.crc32Hex}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase text-slate-500">Decoded Preview</div>
                      <div className="mt-0.5 text-slate-800 truncate">{codecLiveResult.decodedPreview}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: NCM RVQ ABLATION & MULTIMODAL STAGES 1-5 */}
          {activeTab === 'ncm' && (
            <div className="space-y-6" data-testid="lab-panel-ncm">
              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                      Controlled NCM Ablation: Flat 256-State Diffusion vs. 2-Level RVQ + Nested Dropout (.x8D Reloaded)
                    </h3>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      Trained from Step 0 across {report.ncmUnifiedExperiment.trainTotalBytes.toLocaleString()} B (Stages 1–4 incl. PNG &amp; WAV), exported to <span className="font-mono">.x8D v2</span>, training buffers discarded, evaluated strictly from reloaded bytes.
                    </p>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                        <th className="py-2.5 px-4">Model Architecture</th>
                        <th className="py-2.5 px-3 text-right">Eff Weights / DOFs</th>
                        <th className="py-2.5 px-3 text-right">.x8D Disk (Codebook)</th>
                        <th className="py-2.5 px-3 text-right">Held-Out PPL</th>
                        <th className="py-2.5 px-3 text-right">Top-1 Acc</th>
                        <th className="py-2.5 px-3 text-right">Coarse Only → +Fine</th>
                        <th className="py-2.5 px-3 text-right">Avg Steps</th>
                        <th className="py-2.5 px-3 text-right">Reload Diff</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-xs">
                      {report.ncmUnifiedExperiment.models.map((m) => (
                        <tr
                          key={m.modelId}
                          className={m.modelId === 'ncm-rvq-unified' ? 'bg-blue-50/40' : 'bg-white'}
                        >
                          <td className="py-2.5 px-4 font-semibold text-slate-900">{m.label}</td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                            {m.effectiveWeights.toLocaleString()} / {m.trainableDofs.toLocaleString()}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                            {m.physicalX8dBytes.toLocaleString()} B ({m.codebookBytes} B CB)
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-blue-800 whitespace-nowrap">
                            {m.overallHeldOutPpl.toFixed(2)}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-emerald-700 whitespace-nowrap">
                            {m.overallTop1AccPct.toFixed(2)}%
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                            {m.coarseLevelOnlyAccPct.toFixed(1)}% → {m.coarsePlusFineAccPct.toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                            {m.avgStepsPerSpan.toFixed(2)}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-emerald-700 whitespace-nowrap">
                            {m.reloadParityMaxDiff.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Per-Stage Held-Out Curriculum Table */}
              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
                  <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                    Stage 1–5 Multimodal Held-Out Evaluation (Strictly Reloaded From .x8D v2 Disk Checkpoints)
                  </h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                        <th className="py-2.5 px-4">Curriculum Stage</th>
                        <th className="py-2.5 px-3 text-right">Held-Out B</th>
                        <th className="py-2.5 px-3 text-right">Flat Acc (PPL)</th>
                        <th className="py-2.5 px-3 text-right">NCM RVQ Acc (PPL)</th>
                        <th className="py-2.5 px-3 text-right">Δ Acc</th>
                        <th className="py-2.5 px-3 text-right">S1 Exit %</th>
                        <th className="py-2.5 px-3 text-right">Avg Steps</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-xs">
                      {report.ncmUnifiedExperiment.stageBreakdown.map((st) => (
                        <tr key={st.stageId} className={st.stageId === 'stage5-reload-eval' ? 'bg-slate-50 font-semibold' : ''}>
                          <td className="py-2.5 px-4 text-slate-900">{st.stageLabel}</td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700">{st.heldOutBytes}</td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                            {st.flatBaselineTop1Pct.toFixed(1)}% ({st.flatBaselinePpl.toFixed(1)})
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-blue-800 whitespace-nowrap">
                            {st.ncmUnifiedTop1Pct.toFixed(1)}% ({st.ncmUnifiedPpl.toFixed(1)})
                          </td>
                          <td
                            className={`py-2.5 px-3 font-mono tabular-nums text-right whitespace-nowrap ${
                              st.deltaAccPct >= 0 ? 'text-emerald-700' : 'text-amber-700'
                            }`}
                          >
                            {st.deltaAccPct >= 0 ? `+${st.deltaAccPct.toFixed(2)}%` : `${st.deltaAccPct.toFixed(2)}%`}
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800">
                            {st.system1ExitPct.toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900">
                            {st.avgDenoiseSteps.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-700 leading-relaxed space-y-2">
                  <div>
                    <strong className="font-semibold text-slate-900">Stage 5 Fine-Tuning Control ({report.ncmUnifiedExperiment.fineTuneComparison.targetDomain}): </strong>
                    Equal 80-step budget from random scratch = <span className="font-mono">{report.ncmUnifiedExperiment.fineTuneComparison.scratchSameBudgetAccPct}%</span>{' '}
                    (PPL <span className="font-mono">{report.ncmUnifiedExperiment.fineTuneComparison.scratchSameBudgetPpl}</span>) vs. fine-tuning reloaded NCM <span className="font-mono">.x8D</span> checkpoint ={' '}
                    <span className="font-mono font-semibold text-emerald-700">{report.ncmUnifiedExperiment.fineTuneComparison.reloadedFineTuneAccPct}%</span>{' '}
                    (PPL <span className="font-mono">{report.ncmUnifiedExperiment.fineTuneComparison.reloadedFineTunePpl}</span>, Δ{' '}
                    <span className="font-mono">{report.ncmUnifiedExperiment.fineTuneComparison.accGainPct >= 0 ? `+${report.ncmUnifiedExperiment.fineTuneComparison.accGainPct}` : report.ncmUnifiedExperiment.fineTuneComparison.accGainPct}%</span>).
                  </div>
                  <div className="text-[11px] text-slate-600">{report.ncmUnifiedExperiment.honestEvaluationNote}</div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: xAt = 65,536 SEQUENCE ARCHITECTURE */}
          {activeTab === 'xat' && (
            <div className="bg-white border border-slate-200 rounded-lg overflow-hidden" data-testid="lab-panel-xat">
              <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                    xAt = 65,536 Byte-State Sequence Capacity Benchmark ({report.xAtSequenceExperiment.embeddedNeedles} Long-Range Anchors)
                  </h3>
                  <p className="text-[11px] text-slate-500 mt-0.5">{report.xAtSequenceExperiment.clarificationNote}</p>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                      <th className="py-2.5 px-4">Sequence Planner</th>
                      <th className="py-2.5 px-3 text-right">Spans</th>
                      <th className="py-2.5 px-3 text-right">Active Bytes (%)</th>
                      <th className="py-2.5 px-3 text-right">Index Meta</th>
                      <th className="py-2.5 px-3 text-right">Working Buffer</th>
                      <th className="py-2.5 px-3 text-right">Needle Recall</th>
                      <th className="py-2.5 px-3 text-right">Boundary F1</th>
                      <th className="py-2.5 px-4">Empirical Assessment</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 text-xs">
                    {report.xAtSequenceExperiment.rows.map((row) => (
                      <tr
                        key={row.methodId}
                        className={
                          row.methodId === 'adaptive-entropy-cache'
                            ? 'bg-blue-50/40 align-top'
                            : row.methodId === 'fixed-8x8-legacy'
                              ? 'bg-rose-50/40 align-top'
                              : 'bg-white align-top'
                        }
                      >
                        <td className="py-2.5 px-4 font-semibold text-slate-900">{row.label}</td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800">{row.spansCreated}</td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-slate-900 whitespace-nowrap">
                          {row.activeBytesEvaluated.toLocaleString()} B ({row.activeRatioPct}%)
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                          {fmtBytes(row.indexMetadataBytes)}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                          {fmtBytes(row.workingBufferBytes)}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-emerald-700 whitespace-nowrap">
                          {row.needleRetrievalAccPct.toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                          {row.boundaryAlignmentF1Pct.toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-4 text-slate-600 leading-relaxed">{row.verdict}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-700 leading-relaxed">
                <strong className="font-semibold text-slate-900">Hypothesis Audit: </strong>
                {report.xAtSequenceExperiment.honestHypothesisAudit}
              </div>
            </div>
          )}

          {/* TAB 4: MP4 DIRECT-FACTOR EXECUTION */}
          {activeTab === 'mp4' && (
            <div className="bg-white border border-slate-200 rounded-lg overflow-hidden" data-testid="lab-panel-mp4">
              <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                    MP4 Analogy Execution Benchmark ({report.mp4ExecutionExperiment.matrixSpec})
                  </h3>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Active Expert #{report.mp4ExecutionExperiment.activeExpertIndex} routed via real <span className="font-mono">fs.openSync + fs.readSync</span> without OS mmap.
                  </p>
                </div>
                <span className="inline-flex items-center gap-1.5 text-xs font-mono text-emerald-700 font-semibold">
                  <HardDrive className="w-3.5 h-3.5" /> Zero Dense Matrix Materialization (Modes 3–5)
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                      <th className="py-2.5 px-4">Execution Mode</th>
                      <th className="py-2.5 px-3 text-right">Disk File</th>
                      <th className="py-2.5 px-3 text-right">f.readSync Calls</th>
                      <th className="py-2.5 px-3 text-right">Disk Read B</th>
                      <th className="py-2.5 px-3 text-right">Peak Buffer RAM</th>
                      <th className="py-2.5 px-3 text-right">Dense W?</th>
                      <th className="py-2.5 px-3 text-right">Max Abs Diff</th>
                      <th className="py-2.5 px-3 text-right">Cosine</th>
                      <th className="py-2.5 px-3 text-right">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 text-xs">
                    {report.mp4ExecutionExperiment.rows.map((row) => (
                      <tr
                        key={row.modeId}
                        className={row.modeId === 'selective-expert-seek' ? 'bg-blue-50/50' : 'bg-white'}
                      >
                        <td className="py-2.5 px-4 font-semibold text-slate-900">{row.label}</td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                          {fmtBytes(row.diskFileBytes)}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-800 whitespace-nowrap">
                          {row.diskReadsCount}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                          {fmtBytes(row.bytesReadFromDisk)}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right font-semibold text-blue-800 whitespace-nowrap">
                          {fmtBytes(row.peakWorkingBufferBytes)}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-right whitespace-nowrap">
                          <span className={row.materializedDenseMatrix ? 'text-amber-700' : 'text-emerald-700 font-semibold'}>
                            {row.materializedDenseMatrix ? 'Yes' : 'No'}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-emerald-700 whitespace-nowrap">
                          {row.maxAbsDiffVsFactorDense}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                          {row.cosineVsFactorDense.toFixed(6)}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                          {row.latencyMs.toFixed(2)} ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-700 leading-relaxed">
                <strong className="font-semibold text-slate-900">Exact Factor Algebra: </strong>
                {report.mp4ExecutionExperiment.exactAlgebraNote}
              </div>
            </div>
          )}

          {/* TAB 5: REJECTED HYPOTHESES & UPSTREAM GATES */}
          {activeTab === 'hypotheses' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" data-testid="lab-panel-hypotheses">
              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center gap-2">
                  <FlaskConical className="w-4 h-4 text-rose-700" />
                  <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                    Discarded, Rejected &amp; Bounded Hypotheses (Honest Audit)
                  </h3>
                </div>
                <div className="divide-y divide-slate-200 text-xs">
                  {report.discardedOrFailedHypotheses.map((item, idx) => (
                    <div key={idx} className="p-4 space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-semibold text-slate-900">{item.hypothesis}</span>
                        <span
                          className={`font-mono text-[11px] font-semibold uppercase shrink-0 ${
                            item.disposition === 'rejected'
                              ? 'text-rose-700'
                              : item.disposition === 'bounded'
                                ? 'text-amber-700'
                                : 'text-blue-700'
                          }`}
                        >
                          {item.disposition}
                        </span>
                      </div>
                      <p className="text-slate-600 leading-relaxed">{item.measuredEvidence}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-blue-700" />
                  <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-700">
                    Validated Transfer Gates for Upstream Repositories
                  </h3>
                </div>
                <div className="divide-y divide-slate-200 text-xs">
                  {report.upstreamRecommendations.map((rec, idx) => (
                    <div key={idx} className="p-4 space-y-1.5">
                      <div className="font-mono text-[11px] font-semibold text-blue-800">
                        Target: bapXai/{rec.targetRepo}
                      </div>
                      <p className="text-slate-900 font-medium leading-relaxed">{rec.recommendation}</p>
                      <div className="font-mono text-[11px] text-emerald-800 bg-emerald-50/60 border border-emerald-200 rounded px-2.5 py-1.5">
                        Gate: {rec.evidenceGate}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      ) : null}
    </section>
  );
};
