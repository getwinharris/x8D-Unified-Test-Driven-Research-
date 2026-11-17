import React, { useMemo, useState } from 'react';
import {
  AgenticScenarioId,
  CLOUD_MODEL_TIERS,
  CloudModelTierId,
  MEMORY_ARTIFACT_CATALOG,
  MemoryArtifactKind,
  runDualContainerAgenticStep,
  simulateMemoryX8DPacking,
} from '../lib/memoryX8dEngine';
import { formatBytes } from '../lib/architectureMath';

const TIER_IDS: CloudModelTierId[] = ['edge-1t', 'cloud-10t', 'cloud-50t'];

const SCENARIO_OPTIONS: { id: AgenticScenarioId; label: string }[] = [
  { id: 'mcp-code-fix', label: 'MCP Repo Fix (Breakthrough)' },
  { id: 'groot-skill-learn', label: 'Robotics Skill Update' },
  { id: 'personal-workspace-qa', label: 'Private Files & Preferences' },
];

export const MemoryAgentSection: React.FC = () => {
  const [tierId, setTierId] = useState<CloudModelTierId>('cloud-10t');
  const [selectedArtifacts, setSelectedArtifacts] = useState<MemoryArtifactKind[]>([
    'skills-agents-md',
    'mcp-schemas',
    'files-rw-images',
    'user-likes-prefs',
    'verified-tool-steps',
  ]);
  const [hadamardEnabled, setHadamardEnabled] = useState<boolean>(true);
  const [scenarioId, setScenarioId] = useState<AgenticScenarioId>('mcp-code-fix');
  const [mountLocalMemory, setMountLocalMemory] = useState<boolean>(true);

  const selectedTier = CLOUD_MODEL_TIERS[tierId];
  const packSim = useMemo(
    () => simulateMemoryX8DPacking(selectedArtifacts, hadamardEnabled),
    [selectedArtifacts, hadamardEnabled]
  );
  const stepSim = useMemo(
    () => runDualContainerAgenticStep(scenarioId, tierId, mountLocalMemory),
    [scenarioId, tierId, mountLocalMemory]
  );

  const toggleArtifact = (id: MemoryArtifactKind) => {
    setSelectedArtifacts((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  return (
    <section id="memory-agent" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-4xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          08. Dual-Container Architecture · model.x8D (1T–50T) + On-Device memory.x8D
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Personal On-Device <span className="font-mono">memory.x8D</span>: Continual Learning, MCP Skills &amp; Privacy-Safe Promotion
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Instead of brute-forcing giant <span className="font-mono text-slate-800">AGENTS.md</span>,{' '}
          <span className="font-mono text-slate-800">SKILL.md</span>, MCP tool schemas, and private workspace files into
          expensive prompt context on every turn, x8D separates a shared <strong className="font-semibold text-slate-900">1T–50T{' '}
          <span className="font-mono">model.x8D</span></strong> backbone from each user&apos;s sovereign on-device{' '}
          <strong className="font-semibold text-slate-900"><span className="font-mono">memory.x8D</span></strong> binary.
          Both execute together via direct <span className="font-mono text-slate-800">f.seek/read</span> while an explicit
          privacy gate keeps user files &amp; preferences strictly local and promotes only anonymized breakthroughs upstream.
        </p>
      </div>

      {/* 3-tier Cloud + Edge Model Comparison Strip */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        {TIER_IDS.map((id) => {
          const tier = CLOUD_MODEL_TIERS[id];
          const active = tierId === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setTierId(id)}
              data-testid={`tier-card-${id}`}
              className={`text-left rounded-lg border p-4 transition-colors cursor-pointer ${
                active
                  ? 'bg-blue-50/70 border-blue-600 text-slate-900'
                  : 'bg-white border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between gap-2 text-xs font-mono">
                <span className="font-semibold text-blue-700 uppercase">{id.replace('-', ' ')}</span>
                <span className="text-slate-500">0.008 bpw</span>
              </div>
              <div className="mt-1 text-sm font-semibold text-slate-900">{tier.label}</div>
              <div className="mt-2.5 pt-2.5 border-t border-slate-200 grid grid-cols-2 gap-2 text-xs font-mono">
                <div>
                  <div className="text-[10px] text-slate-500 uppercase">model.x8D Disk</div>
                  <div className="font-semibold text-slate-900 tabular-nums">{formatBytes(tier.x8dSubByte0008Bytes)}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500 uppercase">vs BF16 Raw</div>
                  <div className="font-semibold text-emerald-700 tabular-nums">{formatBytes(tier.bf16Bytes)}</div>
                </div>
              </div>
              <div className="mt-2 text-[11px] text-slate-600">{tier.deploymentMode}</div>
            </button>
          );
        })}
      </div>

      {/* Interactive Dual Simulator: Left = memory.x8D Packer, Right = Combined Forward + Continual Learning & Privacy Gate */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="pb-3 border-b border-slate-200 flex items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">On-Device memory.x8D Span Packer</h3>
              <p className="text-xs text-slate-600 mt-0.5">
                Packs skills, MCP schemas, images &amp; likes into <span className="font-mono">0.000–0.255</span> spans.
              </p>
            </div>
            <label className="flex items-center gap-1.5 text-[11px] font-mono text-slate-700 cursor-pointer shrink-0">
              <input
                type="checkbox"
                checked={hadamardEnabled}
                onChange={(e) => setHadamardEnabled(e.target.checked)}
                className="accent-blue-600"
              />
              <span>Hadamard Scale</span>
            </label>
          </div>

          <div className="space-y-2">
            {MEMORY_ARTIFACT_CATALOG.map((artifact) => {
              const checked = selectedArtifacts.includes(artifact.id);
              return (
                <label
                  key={artifact.id}
                  className={`flex items-start gap-2.5 p-2.5 rounded border text-xs cursor-pointer transition-colors ${
                    checked ? 'bg-slate-50 border-slate-300' : 'bg-white border-slate-200 opacity-70'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleArtifact(artifact.id)}
                    className="mt-0.5 accent-blue-600 shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-slate-900">{artifact.label}</span>
                      <span
                        className={`font-mono text-[10px] shrink-0 ${
                          artifact.containsPrivateUserData ? 'text-amber-700' : 'text-emerald-700'
                        }`}
                      >
                        {artifact.containsPrivateUserData ? 'LOCAL PII LOCKED' : 'SKILL ELIGIBLE'}
                      </span>
                    </div>
                    <div className="mt-0.5 text-[11px] font-mono text-slate-500 truncate">
                      {formatBytes(artifact.rawContextBytes)} raw · saves {artifact.promptTokensReplaced.toLocaleString()} prompt tok
                    </div>
                  </div>
                </label>
              );
            })}
          </div>

          <div className="pt-3 border-t border-slate-200 space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-600">Personal memory.x8D File Size</span>
              <span className="font-mono font-bold text-base tabular-nums text-slate-900" data-testid="memory-x8d-bytes">
                {formatBytes(packSim.packedMemoryX8DBytes)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600">Raw Artifacts ({formatBytes(packSim.rawArtifactsBytes)})</span>
              <span className="font-mono font-semibold text-emerald-700 tabular-nums">
                {packSim.compressionRatio.toFixed(1)}× smaller · {packSim.codecExactVerified ? 'Codec 0x01 Exact' : 'Error'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600">Prompt Markdown Tokens Eliminated</span>
              <span className="font-mono font-semibold text-slate-900 tabular-nums">
                {packSim.markdownTokensSavedPerTurn.toLocaleString()} tok / turn
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600">BTC-LLM Outlier SQNR Gain</span>
              <span className="font-mono text-slate-800 tabular-nums">
                {packSim.hadamardOutlierScaleApplied ? `+${packSim.outlierSqnrGainDb.toFixed(1)} dB (Hadamard Active)` : '0.0 dB (Off)'}
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Live Dual-Container Agentic Inference + Continual Learning + Privacy Gate */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="pb-3 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Combined model.x8D + memory.x8D Execution &amp; Privacy Promotion Gate
              </h3>
              <p className="text-xs text-slate-600 mt-0.5">
                Tests continual local tool-step training vs. cloud <span className="font-mono">model.x8D</span> breakthrough sync.
              </p>
            </div>
            <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={mountLocalMemory}
                onChange={(e) => setMountLocalMemory(e.target.checked)}
                className="accent-blue-600"
              />
              <span>Mount Local memory.x8D</span>
            </label>
          </div>

          {/* Scenario selector tabs */}
          <div className="flex flex-wrap gap-1.5 p-1 bg-slate-100 border border-slate-200 rounded-md">
            {SCENARIO_OPTIONS.map((opt) => {
              const active = opt.id === scenarioId;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setScenarioId(opt.id)}
                  className={`flex-1 px-3 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap ${
                    active ? 'bg-white text-slate-900 shadow-xs border border-slate-200' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded p-3.5 space-y-2 text-xs">
            <div className="font-mono text-[11px] uppercase tracking-wider text-slate-500">
              Prompt Input ({selectedTier.id})
            </div>
            <p className="font-medium text-slate-900">{stepSim.userPrompt}</p>
            <p className="text-slate-600 leading-relaxed pt-1 border-t border-slate-200">
              {stepSim.combinedInferenceSummary}
            </p>
          </div>

          {/* MCP + Continual Step Trace */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border border-slate-200 rounded p-3.5 space-y-2">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">
                Direct f.seek/read MCP &amp; Local Train Trace
              </div>
              <ul className="space-y-1.5 text-[11px] font-mono text-slate-700">
                {stepSim.mcpActionsExecuted.map((act, idx) => (
                  <li key={idx} className="truncate">
                    {idx + 1}. {act}
                  </li>
                ))}
              </ul>
              <div className="pt-2 border-t border-slate-200 flex items-center justify-between text-xs font-mono">
                <span className="text-slate-500">Local memory.x8D Delta</span>
                <span className="font-semibold text-slate-900">+{formatBytes(stepSim.localMemoryDeltaBytes)}</span>
              </div>
            </div>

            <div className="border border-slate-200 rounded p-3.5 flex flex-col justify-between space-y-2">
              <div>
                <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">
                  Cloud model.x8D Promotion Gate
                </div>
                <div
                  className={`mt-1.5 text-xs font-mono font-semibold ${
                    stepSim.privacyGateStatus === 'LOCAL_ONLY_LOCKED' ? 'text-amber-700' : 'text-emerald-700'
                  }`}
                  data-testid="privacy-gate-status"
                >
                  {stepSim.privacyGateStatus === 'LOCAL_ONLY_LOCKED'
                    ? '▲ LOCAL_ONLY_LOCKED (Zero Cloud Upload)'
                    : '● ANONYMIZED_BREAKTHROUGH_ELIGIBLE'}
                </div>
                <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">
                  {stepSim.anonymizedBreakthroughSummary}
                </p>
              </div>
              <div className="pt-2 border-t border-slate-200 flex items-center justify-between text-xs font-mono">
                <span className="text-slate-500">Dual Latency · Context Saved</span>
                <span className="font-semibold text-slate-900 tabular-nums">
                  {stepSim.effectiveDualLatencyMs.toFixed(1)} ms · {stepSim.promptMarkdownAvoidedTokens.toLocaleString()} tok
                </span>
              </div>
            </div>
          </div>

          <div className="text-[11px] font-mono text-slate-500 flex flex-wrap items-center justify-between gap-2 pt-1">
            <span>
              memory.x8D Quanta Sample: [{packSim.sampleQuanta.join(', ')}]
            </span>
            <span>
              Active Working RAM ({selectedTier.id}): {formatBytes(selectedTier.activeSpanRamBytes)} (&lt;1 GB ceiling)
            </span>
          </div>
        </div>
      </div>
    </section>
  );
};
