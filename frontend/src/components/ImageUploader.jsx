import { useState, useCallback } from 'react';
import { UploadCloud, FileImage, X } from 'lucide-react';

export default function ImageUploader({ onImageSelected, disabled }) {
    const [isDragging, setIsDragging] = useState(false);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [fileName, setFileName] = useState('');

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
    }, [disabled]);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
        if (disabled) return;
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    }, [disabled]);

    const handleFileInput = useCallback((e) => {
        if (disabled) return;
        const file = e.target.files[0];
        if (file) handleFile(file);
    }, [disabled]);

    const handleFile = (file) => {
        const isNifti = file.name.endsWith('.nii') || file.name.endsWith('.nii.gz');
        const url = isNifti ? null : URL.createObjectURL(file);
        setPreviewUrl(url);
        setFileName(file.name);
        onImageSelected(file, url);
    };

    const clearSelection = (e) => {
        e.stopPropagation();
        setPreviewUrl(null);
        setFileName('');
        onImageSelected(null, null);
    };

    return (
        <div className="w-full">
            <div
                className={`relative rounded-2xl border-2 border-dashed transition-all duration-300
                    ${isDragging ? 'border-brand-400 bg-brand-500/10' : 'border-white/10 bg-gray-900/60 hover:border-brand-500/50 hover:bg-brand-500/[0.03]'}
                    ${disabled ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !disabled && !fileName && document.getElementById('fileUploader').click()}
            >
                <input
                    id="fileUploader"
                    type="file"
                    accept=".nii,.nii.gz,image/*"
                    className="hidden"
                    onChange={handleFileInput}
                    disabled={disabled}
                />

                {!fileName ? (
                    /* Empty state */
                    <div className="flex flex-col items-center justify-center gap-3 py-12 px-6 text-center">
                        <div className="w-16 h-16 rounded-full bg-gray-800/80 flex items-center justify-center border border-white/5 group-hover:scale-110 transition-transform">
                            <UploadCloud className="w-8 h-8 text-brand-400" />
                        </div>
                        <div>
                            <p className="text-base font-medium text-gray-200">Drag & drop your scan</p>
                            <p className="text-xs text-gray-500 mt-1">MRI, NIfTI (.nii / .nii.gz), or image files</p>
                        </div>
                        <span className="mt-1 px-5 py-2 rounded-full bg-brand-500 hover:bg-brand-400 text-white text-sm font-medium shadow-lg shadow-brand-500/20 transition-all">
                            Browse Files
                        </span>
                    </div>
                ) : (
                    /* File loaded */
                    <div className="relative flex flex-col items-center p-4 gap-3">
                        {/* Scanning animation overlay */}
                        {disabled && (
                            <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none z-10">
                                <div className="absolute left-0 right-0 h-0.5 bg-brand-400 shadow-[0_0_16px_4px_rgba(56,189,248,0.7)] animate-scan-line" />
                            </div>
                        )}

                        {/* Image preview */}
                        {previewUrl ? (
                            <img
                                src={previewUrl}
                                alt="Scan preview"
                                className={`w-full max-h-72 object-contain rounded-xl shadow-lg transition-opacity duration-300 ${disabled ? 'opacity-60' : 'opacity-100'}`}
                            />
                        ) : (
                            <div className="flex flex-col items-center gap-3 py-6">
                                <div className="w-20 h-20 rounded-2xl bg-brand-500/10 flex items-center justify-center border border-brand-500/30">
                                    <FileImage className="w-10 h-10 text-brand-400" />
                                </div>
                                <p className="font-semibold text-gray-200 text-sm">{fileName}</p>
                                <p className="text-xs text-gray-500">NIfTI Volume Loaded</p>
                            </div>
                        )}

                        {/* Filename badge + clear */}
                        <div className="flex items-center justify-between w-full px-1">
                            <div className="flex items-center gap-2 text-brand-300 text-xs bg-brand-500/10 px-3 py-1.5 rounded-full border border-brand-500/20">
                                <FileImage className="w-3.5 h-3.5" />
                                <span className="font-medium truncate max-w-[160px]">{fileName}</span>
                            </div>
                            {!disabled && (
                                <button
                                    onClick={clearSelection}
                                    className="bg-gray-800 text-white p-1.5 rounded-full hover:bg-red-500/80 transition-colors"
                                    title="Remove"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
