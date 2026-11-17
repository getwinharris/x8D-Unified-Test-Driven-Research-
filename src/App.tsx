/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useRef, useState } from 'react';
import { TopBar } from './components/TopBar';
import { ExecutiveSummarySection } from './components/ExecutiveSummarySection';
import { SystemDualSection } from './components/SystemDualSection';
import { UnifiedArenaSection } from './components/UnifiedArenaSection';
import { SubBitFrontierSection } from './components/SubBitFrontierSection';
import { RepoAuditSection } from './components/RepoAuditSection';
import { ResearchAdvisorPanel } from './components/ResearchAdvisorPanel';
import { FootprintSimulationInput } from './lib/architectureMath';

export default function App() {
  const advisorInputRef = useRef<HTMLTextAreaElement | null>(null);

  const [simConfig, setSimConfig] = useState<FootprintSimulationInput>({
    scaleId: 'ncm-50m',
    quantId: 'x8d-subbyte-0008',
    executionMode: 'raw-x8d-direct',
    contextBytes: 16384,
    activeLayerSpanRatio: 0.15,
    distributedNodes: 1,
    ncmCoarseBypassRate: 0.65,
  });

  const handleFocusAdvisor = () => {
    const el = document.getElementById('research-advisor');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    setTimeout(() => {
      advisorInputRef.current?.focus();
    }, 300);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <TopBar onFocusAdvisor={handleFocusAdvisor} />

      <main className="flex-1 min-w-0 w-full">
        <ExecutiveSummarySection />
        <SystemDualSection />
        <UnifiedArenaSection />
        <SubBitFrontierSection config={simConfig} onChange={setSimConfig} />
        <RepoAuditSection />
        <ResearchAdvisorPanel config={simConfig} inputRef={advisorInputRef} />
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-[1360px] px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <span>x8D Unified Omni Diffusion–NCM Workbench</span>
          <span className="font-mono">0.000–0.255 Native Bytes · 0.008 bpw .x8D · System 1 + System 2</span>
        </div>
      </footer>
    </div>
  );
}
