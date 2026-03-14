import { useState } from 'react';
import { Eye, Network, X, ZoomIn } from 'lucide-react';

export default function ExplainabilityView({ status, data, selectedImage }) {
    const [lightboxSrc, setLightboxSrc] = useState(null);

    if (status === 'idle') return null;

    const openLightbox = (src) => src && setLightboxSrc(src);
    const closeLightbox = () => setLightboxSrc(null);

    const originalSrc = data?.originalUrl || selectedImage;
    const heatmapSrc = data?.heatmapUrl;

    return (
        <>
            {/* Lightbox overlay */}
            {lightboxSrc && (
                <div
                    className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-4"
                    onClick={closeLightbox}
                >
                    <button
                        onClick={closeLightbox}
                        className="absolute top-4 right-4 text-white bg-gray-800 hover:bg-gray-700 p-2 rounded-full transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                    <img
                        src={lightboxSrc}
                        alt="Full size view"
                        className="max-h-[90vh] max-w-[90vw] object-contain rounded-xl shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    />
                </div>
            )}

            <div className="w-full glass-card rounded-2xl border border-brand-500/20 shadow-[0_0_30px_rgba(14,165,233,0.08)] overflow-hidden">
                {/* Header */}
                <div className="flex items-center gap-3 px-6 py-4 border-b border-white/5">
                    <div className="p-2 bg-brand-500/10 text-brand-400 rounded-xl border border-brand-500/20">
                        <Network className="w-5 h-5" />
                    </div>
                    <div>
                        <h3 className="text-base font-bold text-white">Transformer Attention Map</h3>
                        <p className="text-gray-400 text-xs mt-0.5">Click any image to view full resolution</p>
                    </div>
                </div>

                {/* Images */}
                <div className="grid grid-cols-2 gap-4 p-5">
                    {/* Original */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs font-medium text-gray-400 uppercase tracking-widest">
                            <span>Input (Original)</span>
                            <Eye className="w-3.5 h-3.5" />
                        </div>
                        <div
                            className="relative group aspect-square rounded-xl overflow-hidden bg-black/40 border border-white/5 flex items-center justify-center cursor-zoom-in"
                            onClick={() => openLightbox(originalSrc)}
                        >
                            {originalSrc ? (
                                <>
                                    <img
                                        src={originalSrc}
                                        alt="Original MRI"
                                        className="w-full h-full object-contain rounded-xl transition-transform duration-300 group-hover:scale-105"
                                    />
                                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                                        <ZoomIn className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
                                    </div>
                                </>
                            ) : (
                                <span className="text-gray-600 text-xs">No Image</span>
                            )}
                        </div>
                    </div>

                    {/* Heatmap */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs font-medium text-brand-400 uppercase tracking-widest">
                            <span>Attention Heatmap</span>
                            <Network className="w-3.5 h-3.5" />
                        </div>
                        <div
                            className={`relative group aspect-square rounded-xl overflow-hidden bg-gray-950/80 border border-white/5 flex items-center justify-center ${heatmapSrc ? 'cursor-zoom-in' : ''}`}
                            onClick={() => openLightbox(heatmapSrc)}
                        >
                            {status === 'loading' && (
                                <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900/80 backdrop-blur-md rounded-xl gap-3">
                                    <div className="w-12 h-12 border-4 border-gray-700 border-t-brand-400 rounded-full animate-spin" />
                                    <span className="text-brand-400 text-xs font-medium animate-pulse">Computing Attention…</span>
                                </div>
                            )}

                            {status === 'success' && heatmapSrc && (
                                <>
                                    <img
                                        src={heatmapSrc}
                                        alt="Explainability Heatmap"
                                        className="w-full h-full object-contain rounded-xl transition-transform duration-300 group-hover:scale-105"
                                    />
                                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                                        <ZoomIn className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>

                {/* Clinical insight */}
                {status === 'success' && (
                    <div className="mx-5 mb-5 p-4 rounded-xl bg-brand-500/10 border border-brand-500/20 text-brand-100/90 text-xs leading-relaxed">
                        <strong className="text-brand-300 block mb-1 font-semibold text-sm">Clinical Insight:</strong>
                        Highlighted red/yellow regions show where the Transformer focused attention. In Alzheimer's progression, this typically correlates with hippocampal atrophy and ventricular enlargement — providing transparency for expert medical review.
                    </div>
                )}
            </div>
        </>
    );
}
