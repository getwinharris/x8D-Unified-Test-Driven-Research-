import React, { useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import {
  ALIGNMENT_AUDIT_MATRIX,
  AuditCategory,
  REPO_ISSUES_AND_PRS,
  UNIFIED_EXECUTION_GATES,
} from '../data/researchData';

type FilterTab = 'all' | AuditCategory;

const FILTER_TABS: { id: FilterTab; label: string }[] = [
  { id: 'all', label: 'All Tracked' },
  { id: 'sub1bit-0001', label: 'Sub-1-Bit & 0.001' },
  { id: 'mmap-latent', label: 'Direct f.seek/read' },
  { id: 'ncm-reasoning', label: 'NCM & Mixture' },
  { id: 'training-groot', label: 'Training & #56 QAT' },
  { id: 'tools-runtime', label: 'Tools & PR #74' },
];

export const RepoAuditSection: React.FC = () => {
  const [activeFilter, setActiveFilter] = useState<FilterTab>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const filteredItems = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return REPO_ISSUES_AND_PRS.filter((item) => {
      if (activeFilter !== 'all' && item.category !== activeFilter) return false;
      if (!q) return true;
      return (
        item.title.toLowerCase().includes(q) ||
        item.takeaway.toLowerCase().includes(q) ||
        item.impactOnUnifiedModel.toLowerCase().includes(q) ||
        `#${item.number}`.includes(q)
      );
    });
  }, [activeFilter, searchQuery]);

  return (
    <section id="repo-audit" className="mx-auto max-w-[1360px] px-6 py-12 border-b border-slate-200">
      <div className="max-w-3xl">
        <div className="text-xs font-mono uppercase tracking-wider text-slate-500">
          05. Repo Issues vs. Thread Requirements Audit (bapXai/x8D-Omni-Diffusion)
        </div>
        <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900">
          What Your Repo Issues Say vs. What You Asked For vs. What This Workbench Does
        </h2>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Side-by-side audit reconciling open issues and PR #74 from{' '}
          <span className="font-mono text-slate-800">bapXai/x8D-Omni-Diffusion</span> with your single-model
          requirements in this thread—with every unrequested term and bloat removed.
        </p>
      </div>

      {/* Explicit 3-way Alignment Audit Matrix */}
      <div className="mt-6 bg-white border border-slate-200 rounded-lg overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between gap-4">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-800">
            Three-Way Alignment Audit Matrix
          </h3>
          <span className="text-xs font-mono text-slate-500">1 Unified Model Contract</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/60 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                <th className="py-2.5 px-4">Requirement Axis</th>
                <th className="py-2.5 px-4">Repo Issues Telling Us</th>
                <th className="py-2.5 px-4">You Telling In Thread</th>
                <th className="py-2.5 px-4">Current Workbench State</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-xs">
              {ALIGNMENT_AUDIT_MATRIX.map((row) => (
                <tr key={row.dimension} className="hover:bg-slate-50/70 align-top">
                  <td className="py-3 px-4 font-semibold text-slate-900 min-w-[180px]">
                    <div>{row.dimension}</div>
                    <div
                      className={`mt-1 font-mono text-[10px] uppercase ${
                        row.status === 'aligned'
                          ? 'text-emerald-700'
                          : row.status === 'fixed-this-turn'
                            ? 'text-blue-700'
                            : 'text-amber-700'
                      }`}
                    >
                      {row.status === 'aligned'
                        ? 'Aligned'
                        : row.status === 'fixed-this-turn'
                          ? 'Cleaned & Fixed'
                          : 'Needs #56 QAT Training'}
                    </div>
                  </td>
                  <td className="py-3 px-4 text-slate-700 leading-relaxed min-w-[230px]">{row.repoIssuesSaying}</td>
                  <td className="py-3 px-4 text-slate-700 leading-relaxed min-w-[230px]">{row.userSayingInThread}</td>
                  <td className="py-3 px-4 text-slate-900 leading-relaxed min-w-[240px]">{row.currentWorkbenchDoing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3-phase execution gates */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        {UNIFIED_EXECUTION_GATES.map((phase) => (
          <div key={phase.phase} className="bg-white border border-slate-200 rounded-lg p-4">
            <div className="font-mono text-[11px] font-semibold uppercase text-blue-700">{phase.phase}</div>
            <h3 className="mt-1 text-sm font-semibold text-slate-900">{phase.title}</h3>
            <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">{phase.objective}</p>
            <div className="mt-2.5 pt-2 border-t border-slate-100 text-[11px] font-mono text-emerald-800">
              Gate: {phase.exitGate}
            </div>
          </div>
        ))}
      </div>

      {/* Three-column forensic breakdown of the 0.001 threshold */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-amber-700 font-semibold">UNIT ERROR (ISSUES #50, #58)</span>
            <span className="text-slate-500">x8d_subbyte.py</span>
          </div>
          <h3 className="mt-2 text-base font-semibold text-slate-900">Value Scaling != 0.001 Bit Packing</h3>
          <p className="mt-2 text-xs text-slate-600 leading-relaxed">
            Multiplying a raw byte by <span className="font-mono text-slate-800">0.001</span> maps{' '}
            <span className="font-mono text-slate-800">0..255</span> to{' '}
            <span className="font-mono text-slate-800">0.000..0.255</span> (still 8 bits per byte at I/O), rather than
            compressing stored weights to <span className="font-mono text-slate-800">0.008 bpw</span>.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-rose-700 font-semibold">BLOCK-MEAN LOSS (ISSUES #57, #59)</span>
            <span className="text-slate-500">quantize_hf_safetensors.py</span>
          </div>
          <h3 className="mt-2 text-base font-semibold text-slate-900">1000:1 Collapse Erases Weights</h3>
          <p className="mt-2 text-xs text-slate-600 leading-relaxed">
            Legacy <span className="font-mono text-slate-800">_stream_pack</span> averaged 1000 weights into 1 byte,
            destroying 999 of 1000 degrees of freedom while passing tautological size-only checks (#59).
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-emerald-700 font-semibold">REAL PATH (ISSUES #56, #60, #61)</span>
            <span className="text-slate-500">x8d_arith.py + QAT</span>
          </div>
          <h3 className="mt-2 text-base font-semibold text-slate-900">Kronecker + QAT + x8d_arith (0.008 bpw)</h3>
          <p className="mt-2 text-xs text-slate-600 leading-relaxed">
            Wiring <span className="font-mono text-slate-800">x8d_arith.py</span> (#61) together with Kronecker
            factorization and #56 QAT/STE hits <span className="font-mono text-slate-800">0.008 bpw</span> directly on
            disk with zero <span className="font-mono text-slate-800">{'{-1, 0, +1}'}</span> ternary and no mmap.
          </p>
        </div>
      </div>

      {/* Filter & Search Toolbar */}
      <div className="mt-8 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div
          className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg"
          role="tablist"
          aria-label="Filter repository issues and pull requests"
        >
          {FILTER_TABS.map((tab) => {
            const active = activeFilter === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => setActiveFilter(tab.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  active
                    ? 'bg-white text-slate-900 shadow-xs border border-slate-200'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        <div className="relative w-full lg:w-72">
          <label htmlFor="repo-issue-search" className="sr-only">
            Search issues and pull requests
          </label>
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            id="repo-issue-search"
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter by #id, 0.001, f.seek, NCM..."
            className="w-full rounded-md border border-slate-300 bg-white pl-9 pr-3 py-1.5 text-xs text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none"
          />
        </div>
      </div>

      {/* High-density audit table */}
      <div className="mt-4 bg-white border border-slate-200 rounded-lg overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 bg-slate-50 border-b border-slate-200 text-xs font-mono text-slate-500">
          <span>
            SHOWING <strong className="text-slate-900 tabular-nums">{filteredItems.length}</strong> OF{' '}
            <strong className="text-slate-900 tabular-nums">{REPO_ISSUES_AND_PRS.length}</strong> TRACKED REPO ITEMS
          </span>
          <span>40 OPEN ISSUES · PR #74 OPEN</span>
        </div>

        {filteredItems.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-sm font-medium text-slate-900">No matching repository items found</p>
            <p className="mt-1 text-xs text-slate-500">Clear your search query or switch category tabs.</p>
            <button
              type="button"
              onClick={() => {
                setActiveFilter('all');
                setSearchQuery('');
              }}
              className="mt-3 rounded border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Reset filters
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/70 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                  <th className="py-2.5 px-4 whitespace-nowrap">Ref</th>
                  <th className="py-2.5 px-4 whitespace-nowrap">Status</th>
                  <th className="py-2.5 px-4">Title &amp; Repo Finding</th>
                  <th className="py-2.5 px-4">Required Fix for Unified Model</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-xs">
                {filteredItems.map((item) => {
                  const statusTone =
                    item.state === 'OPEN'
                      ? item.kind === 'PR'
                        ? 'text-blue-700'
                        : 'text-amber-700'
                      : 'text-emerald-700';
                  return (
                    <tr key={`${item.kind}-${item.number}`} className="hover:bg-slate-50/80 transition-colors align-top">
                      <td className="py-3 px-4 font-mono font-semibold text-slate-900 whitespace-nowrap tabular-nums">
                        {item.kind} #{item.number}
                      </td>
                      <td className={`py-3 px-4 font-mono font-medium whitespace-nowrap ${statusTone}`}>
                        {item.state}
                      </td>
                      <td className="py-3 px-4 max-w-xl">
                        <div className="font-semibold text-slate-900">{item.title}</div>
                        <p className="mt-1 text-slate-600 leading-relaxed">{item.takeaway}</p>
                      </td>
                      <td className="py-3 px-4 max-w-md text-slate-700 leading-relaxed">
                        <div>{item.proposedSolution}</div>
                        <div className="mt-1 text-[11px] font-mono text-slate-500">Impact: {item.impactOnUnifiedModel}</div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
};
