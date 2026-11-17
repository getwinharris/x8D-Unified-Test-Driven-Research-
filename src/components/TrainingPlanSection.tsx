import React from 'react';
import { TRAINING_ROADMAP } from '../data/researchData';

export const TrainingPlanSection: React.FC = () => {
  return (
    <section id="training-plan" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          09. Efficiency-First Implementation Roadmap
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Phased Training Plan: From 50M GR00T Prototype to Distributed Omni Scaling
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Following the x8D Issue #68/#71 discipline, every phase enforces an empirical exit gate—eliminating broken
          compression metrics before allocating multi-node GR00T compute.
        </p>
      </div>

      <div className="mt-6 bg-white border border-slate-200 rounded-lg divide-y divide-slate-200">
        {TRAINING_ROADMAP.map((step) => (
          <div key={step.phase} className="p-5 grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
            <div className="lg:col-span-3">
              <div className="text-xs font-mono font-semibold text-blue-700 uppercase tracking-wider">
                {step.phase}
              </div>
              <h3 className="mt-1 text-base font-semibold text-slate-900">{step.title}</h3>
            </div>

            <div className="lg:col-span-5 space-y-2 text-xs text-slate-600 leading-relaxed">
              <p>{step.objective}</p>
              <div>
                <span className="font-mono font-semibold text-slate-800">Dataset: </span>
                <span>{step.datasetSlice}</span>
              </div>
            </div>

            <div className="lg:col-span-4 space-y-2 text-xs leading-relaxed">
              <div className="text-slate-700">
                <span className="font-mono font-semibold text-slate-900">Systems: </span>
                <span>{step.memoryAndComputeStrategy}</span>
              </div>
              <div className="font-mono text-emerald-800 bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5">
                Exit Gate: {step.exitGate}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
