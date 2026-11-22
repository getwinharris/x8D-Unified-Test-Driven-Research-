import React, { useState } from 'react';
import { ArrowRight, Loader2 } from 'lucide-react';
import {
  FootprintSimulationInput,
  MODEL_SCALE_PRESETS,
  QUANTIZATION_REGIMES,
  formatBytes,
  simulateFootprint,
} from '../lib/architectureMath';

interface ResearchAdvisorPanelProps {
  config: FootprintSimulationInput;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
}

const PRESET_QUERIES = [
  'Compare Prompt-to-Output NCM concepts vs. adaptive entropy chunks vs. fixed 8x8 baseline under matched compute inside one unified model.',
  'Design the Issue #56 QAT/STE training schedule so dense FFN/attention weights export cleanly to 0.008 bpw .x8D without #57 block-mean loss.',
  'Detail the C/Rust open("rb") + f.seek/f.read tile loop so a 1T unified model runs System 1 decisions and System 2 loops under 1 GB RAM on CPU/phone.',
];

export const ResearchAdvisorPanel: React.FC<ResearchAdvisorPanelProps> = ({ config, inputRef }) => {
  const [query, setQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [metaInfo, setMetaInfo] = useState<{ model: string; thinkingLevel: string } | null>(null);

  const buildContextSummary = (): string => {
    const sim = simulateFootprint(config);
    const scale = MODEL_SCALE_PRESETS[config.scaleId];
    const quant = QUANTIZATION_REGIMES[config.quantId];
    return [
      `Scale: ${scale.label} (${sim.params.totalParams.toLocaleString()} params, ${scale.nLayers}L, d=${scale.dModel})`,
      `Quantization: ${quant.name} (${sim.effectiveBitsPerWeight} bpw, viable=${sim.forwardPassViable})`,
      `Execution: ${config.executionMode} | Disk=${formatBytes(sim.diskBytes)} | ActiveRAM=${formatBytes(sim.activeWorkingSetRamBytes)} | TileDRAM=${formatBytes(sim.peakAluTileDramBytes)}`,
      `NCM Skip Rate: ${Math.round(config.ncmCoarseBypassRate * 100)}%`,
    ].join('\n');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setError('Enter a specific engineering question or pick one of the research prompts below.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/research/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: trimmed,
          contextSummary: buildContextSummary(),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data?.error || `Request failed (${response.status})`);
        return;
      }
      setAnalysis(data.analysis);
      setMetaInfo({ model: data.model, thinkingLevel: data.thinkingLevel });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Network error reaching the research advisor endpoint.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="research-advisor" className="mx-auto max-w-[1360px] px-6 py-12">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          06. Engineering Advisor (Unified x8D Architecture)
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Interactive High-Thinking Architecture Analysis
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Query the server-side reasoning backend using your active <span className="font-mono">.x8D</span> scale and
          sub-bit parameters as live engineering context.
        </p>
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <form onSubmit={handleSubmit} className="lg:col-span-5 min-w-0 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div>
            <label htmlFor="advisor-query-input" className="block text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">
              Technical Query
            </label>
            <textarea
              ref={inputRef}
              id="advisor-query-input"
              rows={5}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about Issue #56 QAT/STE, x8d_arith frames, Prompt-to-Output NCM, or direct f.seek/read streaming..."
              className="w-full rounded-md border border-slate-300 bg-white p-3 text-xs text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none leading-relaxed"
            />
          </div>

          <div className="space-y-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">Suggested Prompts</div>
            {PRESET_QUERIES.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setQuery(preset)}
                className="w-full text-left rounded border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700 hover:bg-slate-100 hover:border-slate-300 transition-colors cursor-pointer leading-relaxed"
              >
                {preset}
              </button>
            ))}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 rounded-md bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-60 transition-colors cursor-pointer"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <span>Run Architecture Analysis</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        <div className="lg:col-span-7 min-w-0 bg-white border border-slate-200 rounded-lg p-5 min-h-[340px] flex flex-col">
          <div className="flex items-center justify-between gap-2 pb-3 border-b border-slate-200 text-xs font-mono text-slate-500">
            <span>ADVISOR OUTPUT</span>
            {metaInfo ? (
              <span>
                {metaInfo.model} · ThinkingLevel.{metaInfo.thinkingLevel}
              </span>
            ) : (
              <span>Active Context: {MODEL_SCALE_PRESETS[config.scaleId].id}</span>
            )}
          </div>

          {error && (
            <div role="alert" className="mt-4 rounded border border-rose-200 bg-rose-50 p-3.5 text-xs text-rose-800 leading-relaxed">
              <strong className="font-mono">ERROR: </strong>
              {error}
            </div>
          )}

          {loading && (
            <div className="flex-1 flex flex-col items-center justify-center py-12 text-center">
              <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
              <p className="mt-3 text-xs font-medium text-slate-700">
                Synthesizing architectural constraints and memory budgets...
              </p>
            </div>
          )}

          {!loading && !error && !analysis && (
            <div className="flex-1 flex flex-col items-center justify-center py-12 text-center max-w-md mx-auto">
              <p className="text-sm font-medium text-slate-900">Ready for deep systems analysis</p>
              <p className="mt-1.5 text-xs text-slate-500 leading-relaxed">
                Select a prompt or enter your own engineering question on the left. Your active scale,
                sub-bit mode, and NCM parameters are injected automatically.
              </p>
            </div>
          )}

          {!loading && analysis && (
            <div className="mt-4 text-xs sm:text-sm text-slate-800 leading-relaxed whitespace-pre-wrap font-sans overflow-x-auto">
              {analysis}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
