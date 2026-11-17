import React from 'react';

interface TopBarProps {
  onFocusAdvisor: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({ onFocusAdvisor }) => {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1360px] items-center justify-between gap-8 px-6 py-3.5">
        <a
          href="#summary"
          className="font-display text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-blue-600"
        >
          x8D Unified Lab
        </a>

        <nav aria-label="Primary navigation" className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
          <a
            href="#summary"
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Unified Loop
          </a>
          <a
            href="#system-dual"
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            System 1 vs 2
          </a>
          <a
            href="#hf-workbench"
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            HF Test &amp; Train
          </a>
          <a
            href="#unified-arena"
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Sudoku &amp; Snake
          </a>
          <a
            href="#subbit-frontier"
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            0.008 bpw .x8D
          </a>
          <a
            href="#memory-agent"
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            memory.x8D
          </a>
        </nav>

        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={onFocusAdvisor}
            className="rounded-md bg-blue-600 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 cursor-pointer"
          >
            Ask Advisor
          </button>
        </div>
      </div>
    </header>
  );
};
