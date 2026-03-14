import { Brain, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Header({ onHeroUpload }) {
    return (
        <header className="w-full py-12 text-center flex flex-col items-center justify-center relative overflow-hidden">
            {/* Background glow effects */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-brand-500/20 blur-[120px] rounded-full pointer-events-none" />

            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
                className="relative z-10 flex items-center justify-center space-x-4 mb-4"
            >
                <div className="relative flex items-center justify-center w-14 h-14 bg-brand-500/10 rounded-2xl border border-brand-500/20 shadow-[0_0_30px_rgba(14,165,233,0.3)]">
                    <Brain className="w-8 h-8 text-brand-400" />
                    <Sparkles className="absolute -top-2 -right-2 w-5 h-5 text-accent-400 animate-pulse" />
                </div>
                <h1 className="text-6xl md:text-7xl font-extrabold tracking-tight bg-gradient-to-br from-white via-blue-100 to-brand-400 bg-clip-text text-transparent">
                    XAI-TransMed
                </h1>
            </motion.div>

            <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.6 }}
                className="text-gray-300 max-w-3xl mx-auto text-xl leading-relaxed px-4 relative z-10"
            >
                Explainable Transformer for Alzheimer's Prediction. Upload an MRI to get high-accuracy diagnostic insights backed by interpretable heatmaps.
            </motion.p>

            <div className="mt-8 relative z-10 flex items-center space-x-4">
                <a href="#upload-section" className="px-6 py-3 bg-brand-500 hover:bg-brand-400 text-white rounded-full text-lg font-semibold shadow-lg">Try</a>
                <label className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-full cursor-pointer" htmlFor="heroFileInput">Upload</label>
                <input id="heroFileInput" type="file" accept=".nii,.nii.gz,image/*" className="hidden" onChange={(e)=> onHeroUpload && onHeroUpload(e.target.files[0], e.target.files[0] ? URL.createObjectURL(e.target.files[0]) : null)} />
            </div>
        </header>
    );
}
