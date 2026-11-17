import React from 'react';
import { SPATIAL_ENCODING_LITERATURE } from '../data/researchData';

export const SpatialNcmSection: React.FC = () => {
  return (
    <section id="spatial-ncm" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          06. Attention-Based Spatial Encoding &amp; GR00T Alignment
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          Optimizing Latent Representation Efficiency for Multimodal Structural Data
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Training a 50M model from scratch on NVIDIA GR00T robotics trajectories requires avoiding pixel-space quadratic
          attention bottlenecks. Synthesizing 2025–2026 spatial-temporal encoding literature yields a tightly coupled
          architecture combining Video VAE latent patching, 3:1 KDA:MLA memory, Block AttnRes, and NCM coarse-to-fine
          block scheduling.
        </p>
      </div>

      {/* Literature synthesis table */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                <th className="py-3 px-4">Architecture / Paper</th>
                <th className="py-3 px-4">Core Spatial-Latent Mechanism</th>
                <th className="py-3 px-4 whitespace-nowrap">Reported Efficiency</th>
                <th className="py-3 px-4">Role in 50M Hybrid NCM</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-xs">
              {SPATIAL_ENCODING_LITERATURE.map((lit) => (
                <tr key={lit.id} className="hover:bg-slate-50/70 align-top">
                  <td className="py-3.5 px-4 min-w-[200px]">
                    <div className="font-semibold text-slate-900">{lit.paper}</div>
                    <div className="mt-1 font-mono text-[11px] text-slate-500">{lit.venueOrId}</div>
                  </td>
                  <td className="py-3.5 px-4 text-slate-600 leading-relaxed max-w-sm">{lit.mechanism}</td>
                  <td className="py-3.5 px-4 font-mono text-cyan-800 font-medium max-w-[200px] leading-relaxed">
                    {lit.efficiencyGain}
                  </td>
                  <td className="py-3.5 px-4 text-slate-700 leading-relaxed max-w-sm">{lit.roleIn50MHybrid}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* GR00T Multimodal Structural Data Contract */}
      <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="text-xs font-mono text-slate-500">GR00T MODALITY STREAM A</div>
          <h3 className="mt-1.5 text-base font-semibold text-slate-900">Multi-View Spatial-Temporal Vision</h3>
          <p className="mt-2 text-xs text-slate-600 leading-relaxed">
            Ego-centric &amp; workspace RGB streams from <span className="font-mono">GR00T-X-Embodiment-Sim</span> pass
            through a frozen 4×8 spatial-temporal encoder. Cross-attention lets kinematic action queries attend only to
            high-salience object patches rather than uniform background grids.
          </p>
          <div className="mt-4 pt-3 border-t border-slate-200 flex items-center justify-between text-xs font-mono text-slate-500">
            <span>Compression</span>
            <span className="text-slate-900 font-semibold tabular-nums">4x Temporal · 8x Spatial</span>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="text-xs font-mono text-slate-500">GR00T MODALITY STREAM B</div>
          <h3 className="mt-1.5 text-base font-semibold text-slate-900">Proprioception &amp; Action Chunks</h3>
          <p className="mt-2 text-xs text-slate-600 leading-relaxed">
            Joint positions, velocities, torques, and whole-body controller targets map into the shared Uni-LaDiR-style
            continuous latent space while external serialized trajectories remain strict unsigned bytes{' '}
            <span className="font-mono">0..255</span> (Issue #2/#54 compatibility).
          </p>
          <div className="mt-4 pt-3 border-t border-slate-200 flex items-center justify-between text-xs font-mono text-slate-500">
            <span>Recurrent Memory</span>
            <span className="text-slate-900 font-semibold tabular-nums">O(1) KDA Matrix State</span>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="text-xs font-mono text-slate-500">DISTRIBUTED EXECUTION</div>
          <h3 className="mt-1.5 text-base font-semibold text-slate-900">FSDP / ZeRO-2 + NVMe Shard Streaming</h3>
          <p className="mt-2 text-xs text-slate-600 leading-relaxed">
            Each distributed worker streams disjoint GR00T trajectory shards via direct{' '}
            <span className="font-mono">f.seek/read</span> span offsets (no mmap). Optimizer states shard via ZeRO-2 using{' '}
            <span className="font-mono">Muon</span> on 2D hidden weights and <span className="font-mono">AdamW</span> on
            embeddings/gates (Issue #56/#62).
          </p>
          <div className="mt-4 pt-3 border-t border-slate-200 flex items-center justify-between text-xs font-mono text-slate-500">
            <span>Target Efficiency</span>
            <span className="text-slate-900 font-semibold tabular-nums">&gt;= 85% Linear Scaling</span>
          </div>
        </div>
      </div>
    </section>
  );
};
