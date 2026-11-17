import React, { useState } from 'react';
import { REPO_METADATA } from '../data/researchData';
import { formatBytes, formatParams, MODEL_SCALE_PRESETS, computeParameterBreakdown } from '../lib/architectureMath';

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
    sublabel: 'Text · Game · GR00T',
    memoryFootprint: 'Unified 0..255 byte canvas',
    computeRole: 'Zero adapter bridge',
    detail:
      'Streams language, game actions, audio PCM, and GR00T/LeRobot kinematics natively inside the exact same 0.000–0.255 quanta token domain (byte × 0.001, e.g., A=65 → 0.065) into adaptive chunks, NCM concepts, or KDA recurrent spans.',
  },
  {
    id: 'ncm-in-loop',
    label: '02. NCM Iter 1',
    sublabel: 'Coarse Concept Pass',
    memoryFootprint: 'Shared 50M backbone',
    computeRole: 'Gate H_norm ≤ τ_S1',
    detail:
      'Next-Concept Modeling runs directly inside Iteration 1 of x8D’s denoising loop (no separate NCM model), generating a global structural draft and scoring per-block normalized Shannon entropy.',
  },
  {
    id: 'sys1-expert',
    label: '03. System 1 Exit',
    sublabel: 'Iter 1 Commit (~2–5ms)',
    memoryFootprint: '1 pass · zero diff wait',
    computeRole: 'Game & control ticks',
    detail:
      'Low-entropy predictions, real-time game reflexes, robot control frames, and deterministic PR #74 Fraction math exit immediately after Iteration 1 as a fast System 1 expert inside the same weights.',
  },
  {
    id: 'sys2-loop',
    label: '04. System 2 Loop',
    sublabel: 'Iter 2..T Self-Cond + LaDiR',
    memoryFootprint: '10 KDA + 4 MLA · AttnRes',
    computeRole: 'Iterates high-entropy blocks',
    detail:
      'Only ambiguous reasoning, math, or grasp-contact blocks stay inside x8D’s iterative NCM + Diffusion loop with DiffusionGemma self-conditioning carry and Uni-LaDiR continuous latent reasoning.',
  },
  {
    id: 'direct-seek-exec',
    label: '05. Unified .x8D Image',
    sublabel: '0.008 bpw · f.seek/read',
    memoryFootprint: '50.5 KB / ~15 MB RSS',
    computeRole: 'Direct span streaming',
    detail:
      'Packs raw 0.000–0.255 streams (Codec 0x01) and Hadamard-scaled Kronecker + x8d_arith weights at 0.008 bpw (0.001 B/w, zero -1 ternary) inside one direct f.seek/read .x8D image without OS mmap.',
  },
];

export const ExecutiveSummarySection: React.FC = () => {
  const [selectedStageId, setSelectedStageId] = useState<string>(PIPELINE_STAGES[0].id);
  const selectedStage = PIPELINE_STAGES.find((s) => s.id === selectedStageId) ?? PIPELINE_STAGES[0];
  const ncm50mParams = computeParameterBreakdown(MODEL_SCALE_PRESETS['ncm-50m']);

  return (
    <section id="summary" className="mx-auto max-w-[1360px] px-6 pt-10 pb-12 border-b border-slate-200">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-stretch">
        <div className="lg:col-span-7 flex flex-col justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-500 mb-3">
              <span>UNIFIED WORKBENCH &amp; REPORT</span>
              <span aria-hidden="true">·</span>
              <span>{REPO_METADATA.repo}</span>
              <span aria-hidden="true">·</span>
              <span>1-MODEL 0.000–0.255 SYSTEM 1 + 2</span>
            </div>

            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 leading-[1.15] [text-wrap:balance]">
              Unified x8D Omni Diffusion–NCM Hybrid: System 1 &amp; System 2 in One 0.000–0.255 Model
            </h1>

            <p className="mt-4 text-base text-slate-600 leading-relaxed max-w-2xl">
              An interactive architecture workbench and systems audit unifying Next-Concept Modeling (NCM){' '}
              <strong className="font-semibold text-slate-900">inside x8D&apos;s own denoising loop</strong> across language,
              game play, audio, and GR00T robotics in native <span className="font-mono text-sm text-slate-800">0.000–0.255</span> quanta—combining
              a fast 1-step <strong className="font-semibold text-slate-900">System 1 expert</strong> with iterative{' '}
              <strong className="font-semibold text-slate-900">System 2 refinement</strong>, evaluated via side-by-side quantization, perplexity, and HuggingFace byte streams into a{' '}
              <span className="font-mono text-sm text-slate-800">.x8D</span> container.
            </p>
          </div>

          <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4 pt-6 border-t border-slate-200">
            <div>
              <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Core Verdict 01</div>
              <p className="mt-1 text-sm font-medium text-slate-900">NCM Inside x8D&apos;s Loop</p>
              <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                No split dual-model weights: Iteration 1 acts as System 1 for quick reflex/game ticks; high-entropy blocks keep looping inside the exact same KDA/MLA backbone.
              </p>
            </div>
            <div>
              <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Core Verdict 02</div>
              <p className="mt-1 text-sm font-medium text-slate-900">Unified .x8D Binary Image</p>
              <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                Codec 0x01 + Codec 0x03 compress <span className="font-mono">1.64 MB → 2,582 B</span> (634.5× smaller) via direct <span className="font-mono">f.seek/read</span> (no mmap) while avoiding #57&apos;s 1000:1 block-mean trap.
              </p>
            </div>
            <div>
              <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Core Verdict 03</div>
              <p className="mt-1 text-sm font-medium text-slate-900">Live HF Test &amp; Train</p>
              <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                Stream real rows from LeRobot, GR00T, GSM8K, and TinyStories directly into <span className="font-mono">0.000–0.255</span> blocks to verify lossless packing and 50M FSDP sizing.
              </p>
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-4 pb-3 border-b border-slate-200">
              <h2 className="text-sm font-semibold text-slate-900">Unified x8D + NCM Loop Schematic</h2>
              <span className="text-xs font-mono text-slate-500">Select stage</span>
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
          <div className="text-xs font-mono uppercase tracking-wider text-slate-500">50M Baseline Budget</div>
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
          <div className="text-xs font-mono uppercase tracking-wider text-slate-500">x8D GitHub Scope</div>
          <div className="mt-1 flex items-baseline">
            <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">
              {REPO_METADATA.totalIssues} / {REPO_METADATA.totalPRs}
            </span>
            <span className="text-xs font-mono text-slate-500 ml-1.5">ISSUES / PR</span>
          </div>
          <div className="mt-1 text-xs font-mono text-slate-600 tabular-nums">
            {REPO_METADATA.totalFiles} tracked files · PR #74 open
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Sub-Bit Export Target</div>
          <div className="mt-1 flex items-baseline">
            <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">0.008</span>
            <span className="text-xs font-mono text-slate-500 ml-1.5">BITS / WEIGHT</span>
          </div>
          <div className="mt-1 text-xs font-mono text-emerald-700 tabular-nums">
            ● {formatBytes(Math.ceil((ncm50mParams.totalParams * 0.008) / 8))} Target .x8D (Projected)
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-slate-500">Raw .x8D Working Set</div>
          <div className="mt-1 flex items-baseline">
            <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">~15–22</span>
            <span className="text-xs font-mono text-slate-500 ml-1.5">MB RSS</span>
          </div>
          <div className="mt-1 text-xs font-mono text-slate-600 tabular-nums">
            Direct f.seek/read · zero-VRAM (no mmap)
          </div>
        </div>
      </div>
    </section>
  );
};
