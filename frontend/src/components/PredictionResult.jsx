import { Activity, ShieldCheck, AlertTriangle, Layers } from 'lucide-react';

export default function PredictionResult({ status, data, onExplainClick, explainStatus, errorMessage }) {
    if (status === 'idle') {
        return (
            <div className="flex flex-col items-center justify-center h-full min-h-56 rounded-2xl border border-dashed border-white/10 bg-gray-900/40 text-center px-6 py-10 gap-3">
                <Activity className="w-8 h-8 text-gray-600" />
                <p className="text-gray-500 text-sm">Upload a scan to begin analysis</p>
            </div>
        );
    }

    if (status === 'loading') {
        return (
            <div className="flex flex-col items-center justify-center min-h-56 rounded-2xl border border-white/5 bg-gray-900/60 p-8 gap-5">
                <div className="relative w-16 h-16">
                    <div className="absolute inset-0 border-4 border-gray-700 rounded-full" />
                    <div className="absolute inset-0 border-4 border-brand-500 rounded-full border-t-transparent animate-spin" />
                    <Activity className="absolute inset-0 m-auto w-6 h-6 text-brand-400 animate-pulse" />
                </div>
                <div className="text-center">
                    <h3 className="text-lg font-semibold text-white">Analyzing Scan</h3>
                    <p className="text-gray-400 text-sm mt-1">Running Transformer inference…</p>
                </div>
            </div>
        );
    }

    if (status === 'success' && data) {
        const isAD = data.prediction === 'AD';
        const accent = isAD
            ? { bg: 'bg-red-500/5', border: 'border-red-500/20', icon: 'text-red-400' }
            : { bg: 'bg-emerald-500/5', border: 'border-emerald-500/20', icon: 'text-emerald-400' };

        return (
            <div className={`rounded-2xl border backdrop-blur-xl overflow-hidden ${accent.bg} ${accent.border}`}>
                <div className="p-6 space-y-5">
                    {/* Prediction header */}
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className={`p-3 rounded-xl bg-gray-950/50 border border-gray-800 ${accent.icon}`}>
                                {isAD ? <AlertTriangle className="w-6 h-6" /> : <ShieldCheck className="w-6 h-6" />}
                            </div>
                            <div>
                                <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">Prediction</p>
                                <h2 className="text-2xl font-bold text-white mt-0.5">{data.predictionName}</h2>
                            </div>
                        </div>
                        <div className="text-right">
                            <span className="text-3xl font-black text-white">
                                {(data.confidence * 100).toFixed(0)}<span className="text-xl text-gray-400">%</span>
                            </span>
                            <p className="text-xs text-gray-400 uppercase tracking-wider mt-0.5">Confidence</p>
                        </div>
                    </div>

                    {/* Metadata */}
                    <div className="text-xs text-gray-500 border-t border-gray-700/50 pt-4 space-y-0.5">
                        <span>Session: <span className="font-mono tracking-wider">{data.uuid?.split('-')[0]}…</span></span>
                        {data.apiName && <div>API: <span className="font-mono">{data.apiName}</span></div>}
                    </div>

                    {/* Explainability Button */}
                    <button
                        onClick={onExplainClick}
                        disabled={explainStatus === 'loading' || explainStatus === 'success'}
                        className={`w-full flex items-center justify-center gap-2 px-5 py-3 rounded-xl font-medium text-sm transition-all duration-300
                            ${explainStatus === 'loading' || explainStatus === 'success'
                                ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
                                : 'bg-brand-600 hover:bg-brand-500 text-white shadow-[0_0_20px_rgba(2,132,199,0.3)] hover:shadow-[0_0_30px_rgba(14,165,233,0.5)] hover:-translate-y-0.5'
                            }`}
                    >
                        <Layers className={`w-4 h-4 ${explainStatus === 'loading' ? 'animate-spin' : ''}`} />
                        {explainStatus === 'loading' ? 'Generating Map…' :
                            explainStatus === 'success' ? 'Map Generated ✓' :
                                'Generate Explainability Map'}
                    </button>
                </div>
            </div>
        );
    }

    if (status === 'error') {
        return (
            <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-6 text-center space-y-4">
                <div className="flex justify-center">
                    <div className="p-3 rounded-xl bg-gray-950/50 border border-gray-800 text-red-400">
                        <AlertTriangle className="w-6 h-6" />
                    </div>
                </div>
                <div>
                    <h3 className="text-lg font-semibold text-white">Analysis Failed</h3>
                    <p className="text-red-300 text-xs font-mono mt-2 bg-red-900/20 p-3 rounded-lg border border-red-500/30 break-all">
                        {errorMessage || 'The server encountered an error while processing your scan.'}
                    </p>
                </div>
                <button
                    onClick={() => window.location.reload()}
                    className="px-5 py-2 rounded-full bg-gray-800 text-white text-sm hover:bg-gray-700 transition-colors"
                >
                    Try Again
                </button>
            </div>
        );
    }

    return null;
}
