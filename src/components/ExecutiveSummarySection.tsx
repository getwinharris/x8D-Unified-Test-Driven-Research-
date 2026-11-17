import React, { useState } from 'react';
import { REPO_METADATA } from '../data/researchData';
import {
  MODEL_SCALE_PRESETS,
  computeParameterBreakdown,
  formatBytes,
  formatParams,
  simulateFootprint,
} from '../lib/architectureMath';

interface StageDetail {
  id: string;
  label: string;
  sublabel: string;
  memoryFootprint: string;
  computeRole: string;
  detail: string;
}

const PIPELINE_STAGES: StageDetail[] = [
  {
    id: 'omni-stream',
    label: '01. 0–0.255 Bytes',
    sublabel: '8-bit I/O Input',
    memoryFootprint: 'vocab_size = 256',
    computeRole: 'Zero BPE / MASK',
    detail:
      'Every prompt or dataset byte (0..255 across language, code, math, QA, games, audio, and robotics) enters natively via byte × 0.001 into 0.000–0.255 quanta and decodes cleanly back to 8-bit UTF-8 output.',
  },
  {
    id: 'ncm-in-loop',
    label: '02. NCM Iter 1',
    sublabel: 'Prompt→Output Pass',
    memoryFootprint: 'arXiv:2610.14162',
    computeRole: 'Coarse Blueprint',
    detail:
      'Next-Concept Modeling (NCM) runs inside Iteration 1 of x8D’s denoising loop from Prompt to Output, generating a global concept outline and scoring per-span entropy in one unified backbone.',
  },
  {
    id: 'sys1-expert',
    label: '03. S1 Expert',
    sublabel: 'Inside Mixture',
    memoryFootprint: 'Iter 1 Commit (~2–5 ms)',
    computeRole: 'Games · PR #74',
    detail:
      'System 1 lives directly inside the same mixture alongside AR, code, math, and QA—handling real-time decision ticks (Snake, robot actions) and deterministic PR #74 Fraction math in 1 pass.',
  },
  {
    id: 'sys2-loop',
    label: '04. System 2 Loop',
    sublabel: 'Iter 2..T Diffusion',
    memoryFootprint: '3:1 KDA:MLA + AttnRes',
    computeRole: 'Hard Reasoning Spans',
    detail:
      'Ambiguous logic, Sudoku constraints, and multi-step planning refine across Iterations 2..T using shared latent carry before decoding output bytes back to text or control signals.',
  },
  {
    id: 'direct-seek-exec',
    label: '05. 0.008 bpw .x8D',
    sublabel: 'Direct f.seek/read',
    memoryFootprint: '<1 GB RAM @ 1T Scale',
    computeRole: 'No mmap · No -1 Ternary',
    detail:
      'Stores trained parameters toward 0.008 bits per full 8-bit (0.001 B/w) via Kronecker decomposition + QAT (#56) + x8d_arith (#60/#61), read directly from disk via open("rb") + f.seek/read.',
  },
];

export const ExecutiveSummarySection: React.FC = () => {
  const [selectedStageId, setSelectedStageId] = useState<string>(PIPELINE_STAGES[0].id);
  const selectedStage = PIPELINE_STAGES.find((s) => s.id === selectedStageId) ?? PIPELINE_STAGES[0];
  const ncm50mParams = computeParameterBreakdown(MODEL_SCALE_PRESETS['ncm-50m']);
  const oneTEdge = simulateFootprint({
    scaleId: 'x8d-1t',
    quantId: 'x8d-subbyte-0008',
    executionMode: 'raw-x8d-direct',
    contextBytes: 16384,
    activeLayerSpanRatio: 0.15,
    distributedNodes: 1,
    ncmCoarseBypassRate: 0.65,
  });

  return (
    <section id="summary" className="mx-auto max-w-[1360px] px-6 pt-10 pb-12 border-b border-slate-200">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-stretch">
        <div className="lg:col-span-7 min-w-0 flex flex-col justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-500 mb-3">
              <span>SINGLE UNIFIED MODEL ARCHITECTURE</span>
              <span aria-hidden="true">·</span>
              <span>{REPO_METADATA.repo}</span>
              <span aria-hidden="true">·</span>
              <span>0.008 BPW TARGET · NO MMAP</span>
            </div>

            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 leading-[1.15] [text-wrap:balance]">
              One Unified x8D Omni Diffusion–NCM Model: 0.008 bpw Storage &amp; In-Mixture System 1 + 2
            </h1>

            <p className="mt-4 text-base text-slate-600 leading-relaxed max-w-2xl">
              Instead of splitting models across modalities or burning full 48-step passes on every input,{' '}
              <strong className="font-semibold text-slate-900">x8D Omni Diffusion–NCM</strong> unifies Next-Concept
              Modeling (Prompt→Output efficiency), a fast{' '}
              <strong className="font-semibold text-slate-900">System 1 decision expert inside the same mixture</strong>,
              and deliberate System 2 diffusion over default 8-bit inputs (<span className="font-mono text-sm text-slate-800">0.000–0.255</span>).
              Weights target <span className="font-mono text-sm text-slate-800">0.008</span> bits per 8-bit weight in a
              single <span className="font-mono text-sm text-slate-800">.x8D</span> file streamed directly via{' '}
              <span className="font-mono text-sm text-slate-800">f.seek/read</span> under 1 GB RAM.
            </p>
          </div>

          <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4 pt-6 border-t border-slate-200">
            <div>
              <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Unified Mixture</div>
              <p className="mt-1 text-sm font-medium text-slate-900">System 1 Expert Inside Model</p>
              <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                System 1 decision control (Snake, robotics, PR #74 math) runs inside the same weights alongside AR, code, math, and language QA.
              </p>
            </div>
            <div>
              <div className="text-xs font-mono uppercase tracking-wider text-slate-500">0.008 bpw Target</div>
              <p className="mt-1 text-sm font-medium text-slate-900">0.001 Sub-Byte Without -1 Ternary</p>
              <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                Hardware executes 8-bit bytes at I/O (<span className="font-mono">0..255</span>), while Kronecker + QAT (#56) +{' '}
                <span className="font-mono">x8d_arith</span> compress stored weights to <span className="font-mono">0.008 bpw</span>.
              </p>
            </div>
            <div>
              <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Edge Serving</div>
              <p className="mt-1 text-sm font-medium text-slate-900">Direct f.seek/read (No mmap)</p>
              <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                Reads active compressed spans straight from disk into L1/L2 CPU tiles so even 1T-scale edge execution stays below 1 GB RAM.
              </p>
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 min-w-0 bg-white border border-slate-200 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-4 pb-3 border-b border-slate-200">
              <h2 className="text-sm font-semibold text-slate-900">Unified Loop Schematic (Select Stage)</h2>
              <span className="text-xs font-mono text-slate-500">1 Model</span>
            </div>

            <div className="mt-4 space-y-2.5" role="group" aria-label="Interactive pipeline architecture stages">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                {PIPELINE_STAGES.slice(0, 3).map((stage) => {
                  const active = stage.id === selectedStage.id;
                  return (
                    <button
                      key={stage.id}
                      type="button"
                      onClick={() => setSelectedStageId(stage.id)}
                      className={`text-left p-2.5 rounded border transition-colors cursor-pointer ${
                        active
                          ? 'bg-blue-50/70 border-blue-600 text-slate-900'
                          : 'bg-slate-50 border-slate-200 text-slate-700 hover:border-slate-300'
                      }`}
                    >
                      <div className="text-[11px] font-mono font-semibold text-blue-700 truncate">{stage.label}</div>
                      <div className="text-xs font-medium text-slate-900 mt-0.5 truncate">{stage.sublabel}</div>
                      <div className="text-[11px] font-mono text-slate-500 mt-1 truncate">{stage.computeRole}</div>
                    </button>
                  );
                })}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {PIPELINE_STAGES.slice(3).map((stage) => {
                  const active = stage.id === selectedStage.id;
                  return (
                    <button
                      key={stage.id}
                      type="button"
                      onClick={() => setSelectedStageId(stage.id)}
                      className={`text-left p-2.5 rounded border transition-colors cursor-pointer ${
                        active
                          ? 'bg-blue-50/70 border-blue-600 text-slate-900'
                          : 'bg-slate-50 border-slate-200 text-slate-700 hover:border-slate-300'
                      }`}
                    >
                      <div className="text-[11px] font-mono font-semibold text-cyan-700 truncate">{stage.label}</div>
                      <div className="text-xs font-medium text-slate-900 mt-0.5 truncate">{stage.sublabel}</div>
                      <div className="text-[11px] font-mono text-slate-500 mt-1 truncate">{stage.memoryFootprint}</div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-slate-200 bg-slate-50 rounded p-3.5">
            <div className="flex items-center justify-between gap-2 text-xs font-mono text-slate-600">
              <span className="font-semibold text-slate-900">{selectedStage.label}</span>
              <span>{selectedStage.memoryFootprint}</span>
            </div>
            <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">{selectedStage.detail}</p>
          </div>
        </div>
      </div>

      {/* Full-width 4-column telemetry strip below hero split */}
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 border border-slate-200 bg-white rounded-lg divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-slate-500">50M Baseline Scale</div>
          <div className="mt-1 flex items-baseline">
            <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">
              {formatParams(ncm50mParams.totalParams)}
            </span>
            <span className="text-xs font-mono text-slate-500 ml-1.5">PARAMS</span>
          </div>
          <div className="mt-1 text-xs font-mono text-slate-600 tabular-nums">
            14L · d=512 · 256 byte I/O
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Sub-Bit Disk Target</div>
          <div className="mt-1 flex items-baseline">
            <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">0.008</span>
            <span className="text-xs font-mono text-slate-500 ml-1.5">BITS / WEIGHT</span>
          </div>
          <div className="mt-1 text-xs font-mono text-emerald-700 tabular-nums">
            {formatBytes(Math.ceil((ncm50mParams.totalParams * 0.008) / 8))} (50M) · Zero -1 ternary
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-slate-500">1T Edge .x8D Target</div>
          <div className="mt-1 flex items-baseline">
            <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">
              {formatBytes(oneTEdge.diskBytes)}
            </span>
            <span className="text-xs font-mono text-slate-500 ml-1.5">DISK</span>
          </div>
          <div className="mt-1 text-xs font-mono text-slate-600 tabular-nums">
            {formatBytes(oneTEdge.activeWorkingSetRamBytes)} active RAM (&lt;1 GB)
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Direct Runtime Path</div>
          <div className="mt-1 flex items-baseline">
            <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">f.seek</span>
            <span className="text-xs font-mono text-slate-500 ml-1.5">READ</span>
          </div>
          <div className="mt-1 text-xs font-mono text-slate-600 tabular-nums">
            Compressed .x8D · Zero OS mmap
          </div>
        </div>
      </div>
    </section>
  );
};
