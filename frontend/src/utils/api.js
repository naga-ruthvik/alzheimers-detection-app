const API_BASE_URL = ''; // Proxied via Vite

export const uploadAndPredict = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || response.statusText);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Prediction failed:', error);
    return { status: 'error', message: error.message };
  }
};

export const uploadAndPredictExplain = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`/predict_explain_v2`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || response.statusText);
    }

    const data = await response.json();

    // Normalize to shape expected by the UI used for uploadAndPredict
    if (data.status === 'success' && data.data) {
      const d = data.data;
      const labelMap = {
        'CN': 'Cognitively Normal',
        'AD': "Alzheimer's Disease",
      };

      return {
        status: 'success',
        data: {
          prediction: d.prediction,
          predictionName: labelMap[d.prediction] || d.prediction,
          uuid: d.uuid,
          confidence: d.confidence ?? 0,
          heatmapUrl: d.explainability_image_url || null,
          originalUrl: null,
          apiName: '/predict_explain_v2'
        }
      };
    }

    return { status: 'error', message: data.message || 'Unknown error' };
  } catch (error) {
    console.error('Predict+Explain failed:', error);
    return { status: 'error', message: error.message };
  }
};

export const fetchExplainabilityHeatmap = async (uuid) => {
  // In the real implementation, the heatmap URL is already returned in the predict response.
  // However, if we want to follow the existing UI flow where we fetch it later:
  return {
    status: 'success',
    data: {
      // Assuming the backend returns the URL in the prediction step for now,
      // but if the UI expects a separate call, we'd need a separate endpoint.
      // For now, we'll assume the URL is already partially known or we re-fetch prediction metadata.
      // Let's modify the UI to use the URL from prediction if available.
    }
  };
};
