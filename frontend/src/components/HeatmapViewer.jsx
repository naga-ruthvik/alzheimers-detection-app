function HeatmapViewer({ heatmap }) {

    if (!heatmap) return null;
  
    return (
      <div className="bg-white shadow-lg rounded-xl p-6 mt-6">
  
        <h2 className="text-xl font-semibold text-center">
          Heatmap Explanation
        </h2>
  
        <img
          src={heatmap}
          className="mt-4 rounded-lg"
        />
  
      </div>
    );
  }
  
  export default HeatmapViewer;