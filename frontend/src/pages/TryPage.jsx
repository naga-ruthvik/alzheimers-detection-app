import { useState } from 'react';
import { FlaskConical } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import PredictionResult from '../components/PredictionResult';
import ExplainabilityView from '../components/ExplainabilityView';
import { uploadAndPredict, uploadAndPredictExplain } from '../utils/api';

export default function TryPage({ onNavigate }) {
  const [selectedImageFile, setSelectedImageFile] = useState(null);
  const [selectedImageUrl, setSelectedImageUrl] = useState(null);

  const [predictionStatus, setPredictionStatus] = useState('idle');
  const [predictionData, setPredictionData] = useState(null);

  const [apiMode] = useState('new');

  const [explainStatus, setExplainStatus] = useState('idle');
  const [explainData, setExplainData] = useState(null);

  const [errorMessage, setErrorMessage] = useState(null);

  const handleImageSelected = async (file, url) => {
    setSelectedImageFile(file);
    setSelectedImageUrl(url);

    setPredictionData(null);
    setExplainData(null);
    setExplainStatus('idle');
    setErrorMessage(null);

    if (!file) {
      setPredictionStatus('idle');
      return;
    }

    try {
      setPredictionStatus('loading');
      const response =
        apiMode === 'new'
          ? await uploadAndPredictExplain(file)
          : await uploadAndPredict(file);

      if (response.status === 'success') {
        const apiName =
          response.data?.apiName ||
          (apiMode === 'new' ? '/predict_explain_v2' : '/predict');
        setPredictionData({ ...response.data, apiName });
        setPredictionStatus('success');
      } else {
        setErrorMessage(response.message || 'Unknown error occurred');
        setPredictionStatus('error');
      }
    } catch (error) {
      console.error(error);
      setErrorMessage(error.message);
      setPredictionStatus('error');
    }
  };

  const handleExplainClick = async () => {
    if (!predictionData?.heatmapUrl) return;

    setExplainStatus('loading');
    setTimeout(() => {
      setExplainData({
        heatmapUrl: predictionData.heatmapUrl,
        originalUrl: predictionData.originalUrl,
      });
      setExplainStatus('success');
    }, 500);
  };

  return (
    /* Full height container, no scroll intended */
    <div className="min-h-screen bg-transparent text-slate-50 flex flex-col">
      <main className="flex-1 flex flex-col pt-20 pb-6 px-4 sm:px-6 lg:px-8">

        {/* Page title row */}
        <div className="flex items-center gap-3 mb-6 max-w-7xl mx-auto w-full">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-xs font-mono tracking-widest uppercase">
            <FlaskConical className="w-3.5 h-3.5" />
            Analysis Session
          </div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            MRI Scan Analysis
          </h1>
        </div>

        {/* Two-column grid */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-7xl mx-auto w-full">

          {/* ── LEFT: Input ─────────────────────────────────── */}
          <div className="flex flex-col gap-4">
            <div className="glass-card rounded-2xl border border-white/5 p-6">
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
                Upload Scan
              </h2>
              <ImageUploader
                onImageSelected={handleImageSelected}
                disabled={predictionStatus === 'loading'}
              />
            </div>

            {/* Instructions / hints */}
            <div className="rounded-xl border border-white/5 bg-gray-900/40 p-5 text-xs text-gray-500 leading-relaxed space-y-2">
              <p className="font-semibold text-gray-400 text-sm">How it works</p>
              <p>1. Upload an MRI brain scan (NIfTI or image format).</p>
              <p>2. The Transformer model runs inference and returns a prediction.</p>
              <p>3. Click <strong className="text-brand-400">Generate Explainability Map</strong> to see the attention heatmap.</p>
            </div>
          </div>

          {/* ── RIGHT: Results ──────────────────────────────── */}
          <div className="flex flex-col gap-4 min-h-0">

            {/* Prediction result card */}
            <PredictionResult
              status={predictionStatus}
              data={predictionData}
              onExplainClick={handleExplainClick}
              explainStatus={explainStatus}
              errorMessage={errorMessage}
            />

            {/* Explainability view appears here when triggered */}
            {explainStatus !== 'idle' && (
              <ExplainabilityView
                status={explainStatus}
                data={explainData}
                selectedImage={selectedImageUrl}
              />
            )}
          </div>

        </div>
      </main>
    </div>
  );
}
