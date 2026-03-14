import { useState } from "react";

function UploadSection({ onFileSelect, onPredict }) {

  const [preview, setPreview] = useState(null);

  const handleChange = (e) => {
    const file = e.target.files[0];

    onFileSelect(file);

    setPreview(URL.createObjectURL(file));
  };

  return (
    <div className="bg-white shadow-lg rounded-xl p-6">

      <h2 className="text-xl font-semibold mb-4 text-center">
        Upload MRI Scan
      </h2>

      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="mb-4"
      />

      {preview && (
        <img
          src={preview}
          className="rounded-lg mb-4"
        />
      )}

      <button
        onClick={onPredict}
        className="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700"
      >
        Predict
      </button>

    </div>
  );
}

export default UploadSection;