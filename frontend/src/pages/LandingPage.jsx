import { useEffect, useRef } from 'react';
import { Brain, Zap, Shield, Search, ArrowRight, ChevronRight, Activity, Cpu, Layers } from 'lucide-react';
import Footer from '../components/Footer';
import FeatureCard from '../components/FeatureCard';

const statItems = [
  { value: '94.7%', label: 'Classification Accuracy', icon: Activity },
  { value: '< 2s',  label: 'Inference Time',           icon: Zap },
  { value: '4-Class', label: 'Dementia Staging',        icon: Layers },
  { value: 'XAI',   label: 'Fully Interpretable',       icon: Cpu },
];

const steps = [
  {
    num: '01',
    title: 'Upload MRI Scan',
    desc: "Drop a raw T1-weighted brain MRI image into the secure uploader.",
    color: 'from-brand-600 to-brand-400',
  },
  {
    num: '02',
    title: 'AI Analysis',
    desc: "Our transformer-based model classifies the scan across four Alzheimer's stages in under 2 seconds.",
    color: 'from-accent-600 to-accent-400',
  },
  {
    num: '03',
    title: 'Interpret Results',
    desc: 'Visualise Grad-CAM attention heatmaps highlighting the exact brain regions driving the prediction.',
    color: 'from-emerald-600 to-emerald-400',
  },
];

/** Subtle cursor-following radial glow that sits inside the hero */
function CursorGlow() {
  const glowRef = useRef(null);

  useEffect(() => {
    const el = glowRef.current;
    if (!el) return;
    const onMove = (e) => {
      const rect = el.parentElement.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      el.style.transform = `translate(${x - 300}px, ${y - 300}px)`;
    };
    window.addEventListener('mousemove', onMove, { passive: true });
    return () => window.removeEventListener('mousemove', onMove);
  }, []);

  return (
    <div
      ref={glowRef}
      className="pointer-events-none absolute w-[600px] h-[600px] rounded-full"
      style={{
        background: 'radial-gradient(circle, rgba(14,165,233,0.08) 0%, transparent 70%)',
        transition: 'transform 0.12s ease-out',
        top: 0,
        left: 0,
      }}
    />
  );
}

export default function LandingPage({ onNavigate }) {
  return (
    <div className="flex flex-col min-h-screen bg-transparent text-slate-50">

      {/* ── Hero ── */}
      <section className="relative flex flex-col items-center justify-center text-center min-h-screen px-6 pt-28 pb-20 overflow-hidden">
        <CursorGlow />

        {/* Headline */}
        <h1 className="animate-in fade-in slide-in-from-bottom-4 duration-700 delay-75 text-5xl sm:text-6xl lg:text-7xl font-extrabold leading-tight tracking-tight max-w-4xl mb-6">
          Explainable AI for{' '}
          <span className="bg-gradient-to-r from-brand-400 via-brand-300 to-accent-400 bg-clip-text text-transparent">
            Alzheimer's Diagnosis
          </span>
        </h1>

        {/* Subheading */}
        <p className="animate-in fade-in slide-in-from-bottom-4 duration-700 delay-150 text-lg sm:text-xl text-slate-400 max-w-2xl mb-10 leading-relaxed">
          Upload a brain MRI scan and receive an instant dementia stage classification —
          with transparent, clinician-readable attention heatmaps.
        </p>

        {/* CTAs */}
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-700 delay-200 flex flex-col sm:flex-row gap-4 items-center mb-20">
          <button
            id="hero-cta-try"
            onClick={() => onNavigate('try')}
            className="group flex items-center gap-2 px-8 py-4 rounded-xl bg-brand-500 hover:bg-brand-400 active:bg-brand-600 text-white font-semibold text-base transition-all duration-200 shadow-lg shadow-brand-500/30 hover:shadow-brand-400/40 hover:-translate-y-0.5"
          >
            Try It Now
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
          <a
            href="#how-it-works"
            className="flex items-center gap-1 text-slate-400 hover:text-white text-sm font-medium transition-colors"
          >
            How it works <ChevronRight className="w-4 h-4" />
          </a>
        </div>

        {/* Brain orb */}
        <div className="animate-in fade-in zoom-in-75 duration-1000 delay-500 relative">
          <div className="absolute inset-0 rounded-full bg-brand-500/20 blur-3xl scale-150" />
          <div className="relative flex items-center justify-center w-36 h-36 sm:w-44 sm:h-44 rounded-full bg-gray-900/80 border border-brand-500/20 shadow-2xl ring-1 ring-brand-500/10">
            <Brain className="w-20 h-20 sm:w-26 sm:h-26 text-brand-400 opacity-90" />
          </div>
          {/* Orbiting ring */}
          <div
            className="absolute inset-[-18px] rounded-full border border-dashed border-brand-500/20"
            style={{ animation: 'spin 18s linear infinite' }}
          />
          <div
            className="absolute inset-[-36px] rounded-full border border-dashed border-accent-500/10"
            style={{ animation: 'spin 28s linear infinite reverse' }}
          />
        </div>
      </section>

      {/* ── Stats Bar ── */}
      <section className="py-14 border-y border-gray-800/60 bg-gray-900/40 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          {statItems.map(({ value, label, icon: Icon }) => (
            <div key={label} className="flex flex-col items-center gap-2 group">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-brand-500/10 border border-brand-500/15 group-hover:bg-brand-500/20 transition-all duration-300 mb-1">
                <Icon className="w-5 h-5 text-brand-400 opacity-80" />
              </div>
              <span className="text-3xl font-extrabold text-white tracking-tight">{value}</span>
              <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section className="py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">Built for Clinical Transparency</h2>
            <p className="text-slate-400 max-w-xl mx-auto">
              Every design decision prioritises interpretability alongside accuracy.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard
              icon={Zap}
              title="High Precision"
              description="Powered by advanced Transformer architectures fine-tuned on clinical MRI datasets."
              colorClass="bg-brand-500/10 text-brand-400"
            />
            <FeatureCard
              icon={Shield}
              title="Secure Processing"
              description="Scans are processed with enterprise-grade security. No data is stored post-inference."
              colorClass="bg-emerald-500/10 text-emerald-400"
            />
            <FeatureCard
              icon={Search}
              title="Full Interpretability"
              description="Generate attention heatmaps to understand exactly which brain regions influenced the diagnosis."
              colorClass="bg-accent-500/10 text-accent-400"
            />
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section id="how-it-works" className="py-24 px-6 bg-gray-900/30 scroll-mt-20">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">How It Works</h2>
            <p className="text-slate-400 max-w-xl mx-auto">Three steps from raw scan to explainable diagnosis.</p>
          </div>

          <div className="relative flex flex-col gap-0">
            {/* Vertical connector line */}
            <div className="absolute left-[2.25rem] top-10 bottom-10 w-px bg-gradient-to-b from-brand-500/60 via-accent-500/40 to-emerald-500/30 hidden md:block" />

            {steps.map(({ num, title, desc, color }) => (
              <div key={num} className="flex gap-6 md:gap-8 items-start pb-12 last:pb-0 group">
                {/* Number badge */}
                <div className={`relative flex-shrink-0 w-[4.5rem] h-[4.5rem] rounded-2xl bg-gradient-to-br ${color} flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform duration-300`}>
                  <span className="text-xl font-extrabold text-white/90 font-mono">{num}</span>
                </div>
                {/* Content */}
                <div className="flex-1 pt-3">
                  <h3 className="text-lg font-bold text-white mb-2">{title}</h3>
                  <p className="text-slate-400 leading-relaxed text-sm max-w-lg">{desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Final CTA */}
          <div className="text-center mt-16">
            <button
              id="how-cta-try"
              onClick={() => onNavigate('try')}
              className="group inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-brand-500 hover:bg-brand-400 active:bg-brand-600 text-white font-semibold text-base transition-all duration-200 shadow-lg shadow-brand-500/30 hover:shadow-brand-400/40 hover:-translate-y-0.5"
            >
              Start Analysing
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
