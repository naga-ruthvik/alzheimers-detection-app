function ResultCard({ prediction, onExplain }) {

    if (!prediction) return null;
  
    const color =
      prediction === "AD"
        ? "text-red-600"
        : "text-green-600";
  
    return (
      <div className="bg-white shadow-lg rounded-xl p-6 mt-6 text-center">
  
        <h2 className="text-xl font-semibold">
          Prediction Result
        </h2>
  
        <p className={`text-3xl font-bold mt-3 ${color}`}>
          {prediction}
        </p>
  
        <button
          onClick={onExplain}
          className="mt-4 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
        >
          Show Explainability
        </button>
  
      </div>
    );
  }
  
  export default ResultCard;