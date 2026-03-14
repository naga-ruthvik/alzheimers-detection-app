import { Brain, ArrowRight, ArrowLeft } from 'lucide-react';

export default function Navbar({ currentPage, onNavigate }) {
  return (
    <nav className="fixed top-0 inset-x-0 z-50">
      {/* Glass bar */}
      <div className="mx-4 mt-3 rounded-2xl bg-gray-950/60 backdrop-blur-2xl border border-white/[0.07] shadow-[0_4px_32px_rgba(0,0,0,0.4)]">
        <div className="max-w-7xl mx-auto px-5 h-14 flex items-center justify-between">

          {/* ── Logo ── */}
          <button
            onClick={() => onNavigate?.('landing')}
            className="flex items-center gap-2.5 group cursor-pointer shrink-0"
            id="nav-logo"
          >
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-brand-500/15 border border-brand-500/25 shadow-[0_0_10px_rgba(14,165,233,0.15)] group-hover:bg-brand-500/25 group-hover:shadow-[0_0_16px_rgba(14,165,233,0.25)] transition-all duration-300">
              <Brain className="w-4 h-4 text-brand-400" />
            </div>
            <span className="font-bold text-[15px] tracking-tight text-white group-hover:text-brand-300 transition-colors duration-200">
              XAI-TransMed
            </span>
          </button>

          {/* ── Right actions ── */}
          <div className="flex items-center">
            {currentPage === 'landing' && (
              <button
                id="nav-cta-try"
                onClick={() => onNavigate?.('try')}
                className="group flex items-center gap-1.5 bg-brand-500 hover:bg-brand-400 active:bg-brand-600 text-white px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 shadow-md shadow-brand-500/30 hover:shadow-brand-400/40 hover:-translate-y-px select-none"
              >
                Try It Now
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform duration-200" />
              </button>
            )}

            {currentPage === 'try' && (
              <button
                onClick={() => onNavigate?.('landing')}
                className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-white font-medium transition-colors duration-200 px-3 py-2 rounded-xl hover:bg-white/5 select-none"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Home
              </button>
            )}
          </div>

        </div>
      </div>
    </nav>
  );
}